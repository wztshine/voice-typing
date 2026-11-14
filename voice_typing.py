# -*- coding: utf-8 -*-
"""
本地语音输入法 (FireRedASR2-AED)
按住热键说话 -> 松开 -> 识别结果自动输入到当前光标处
"""
import argparse
import atexit
import ctypes
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback

import numpy as np
import sounddevice as sd
import soundfile as sf
import pyperclip
from pynput import keyboard

import hotword
import postprocess

# 识别库每次识别都用 INFO 级别打印 VAD/ASR/标点中间结果:
# 白白做字符串格式化, pythonw 下还往一个失效的 stderr 写, 压到 WARNING 只留报错
logging.getLogger("fireredasr2s").setLevel(logging.WARNING)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
TMP_WAV = os.path.join(BASE, "logs", "uttr")  # 实际文件名带时间戳, 避免并发覆盖
LOG = os.path.join(BASE, "logs", "voice_typing.log")
CONFIG = os.path.join(BASE, "config.json")

CFG = dict(
    hotkey="ctrl_r",               # 按住说话 (改为 f9 / scroll_lock / right_ctrl ...)
    toggle_key="f10",              # 按一下开始, 再按一下结束(适合长段口述); "none" 关闭
    sample_rate=16000,
    min_sec=0.35,
    max_sec=120,
    device=None,          # None = 系统默认麦克风
    use_half=True,        # FP16 推理 (RTX 2060 支持, 更快更省显存)
    enable_vad=True,
    enable_punc=True,
    paste=True,           # True: 剪贴板粘贴(快, 中文最稳); False: 逐字模拟输入
    restore_clipboard=True,
    beep=True,
    mute_speaker=True,    # 录音期间静音扬声器(防幻听: 音乐/视频/提示音被麦克风收进去识别成字), 结束即恢复
    hotword_bonus=2.0,    # 热词偏置强度(0 关闭). 4~6 更强势但可能误插
    vad_min_silence_ms=600,    # 停顿超过多久才认为一句结束(默认 0.6s)
    vad_merge_silence_ms=1400, # 两段语音间隔小于多久就合并成一段(默认 1.4s)
    vad_max_speech_ms=20000,   # 单段最长, 超过会强制切开
    join_gap_ms=2500,          # 段间停顿小于多久合并成逗号(0 关闭)
    show_ui=True,              # 屏幕底部浮窗(录音状态 + 实时文字)
    live_preview=False,        # 长句边说边显示识别草稿(费显卡, 只要状态反馈就关着)
    live_after_sec=3.0,        # 说满多少秒后开始出草稿(短句不触发, 不拖慢)
    live_interval_ms=1200,     # 草稿刷新间隔
    live_max_sec=10.0,         # 草稿最多看最近多少秒音频
    beam_size=3,               # 解码搜索宽度: 1 最快(约快 2 倍), 3 更准
    postprocess=True,          # 输出后处理总开关(规则改写, 不占显卡)
    filler_removal=True,       # 去口语冗余: 嗯/那个/重复字
    number_norm=True,          # 数字/日期/金额/百分比转阿拉伯数字
    trailing_punct="auto",     # 末尾标点: auto=短句去/长句留; strip=总去; keep=总留
    trailing_punct_max_len=15, # auto 模式: 不超过这个字数的句子, 末尾标点去掉
    trailing_punct_keep="？！…～~",  # 这些符号承载含义/语气, 永不删; 设为 "" 则全去
    uppercase_abbr=["po"],     # 这些独立出现的英文缩写统一大写
    diagnose=False,            # 诊断: 记录 VAD 分段/逐段文本, 并保留最后一段录音 logs/last.wav
    fast_load=True,            # 有 model.fp16.pth.tar 时用 fp16+mmap 加载(峰值内存 ~9GB -> ~2GB)
)


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    if sys.stdout is not None:
        print(line, flush=True)
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > 3 * 1024 * 1024:
            os.replace(LOG, LOG + ".1")
    except Exception:
        pass
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_config():
    """读取 config.json 覆盖默认配置(热键等)

    只接受 CFG 里已经定义过的键(外加 postprocess 的 latin_keep_case),
    这样新增选项不会因为忘记加白名单而静默失效; 不认识的键会写进日志提醒。
    """
    if not os.path.exists(CONFIG):
        return
    try:
        import json
        with open(CONFIG, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        log("读取 config.json 失败, 使用默认配置:\n" + traceback.format_exc())
        return
    if not isinstance(data, dict):
        log("config.json 内容不是对象, 已忽略")
        return
    allowed = set(CFG) | {"latin_keep_case"}
    ignored = sorted(k for k in data if k not in allowed)
    for k in allowed:
        if k in data:
            CFG[k] = data[k]
    if ignored:
        log("config.json 里有未知配置项, 已忽略: " + ", ".join(ignored))
    # 数字项写错时回退默认值, 避免运行时才崩(如 max_sec 写成字符串)
    for k, default in (("sample_rate", 16000), ("max_sec", 120), ("beam_size", 3),
                       ("vad_min_silence_ms", 600), ("vad_merge_silence_ms", 1400),
                       ("vad_max_speech_ms", 20000), ("join_gap_ms", 2500),
                       ("live_interval_ms", 1200)):
        if k in data:
            low = 0 if k == "join_gap_ms" else 1   # join_gap_ms=0 表示关闭合并
            CFG[k] = max(low, _as_int(data[k], default, k))
    for k, default in (("min_sec", 0.35), ("hotword_bonus", 2.0),
                       ("live_after_sec", 3.0), ("live_max_sec", 10.0),
                       ("trailing_punct_max_len", 15.0)):
        if k in data:
            CFG[k] = _as_float(data[k], default, k)
    if "trailing_punct" in data:
        tp = str(data["trailing_punct"]).strip().lower()
        if tp in ("auto", "strip", "keep"):
            CFG["trailing_punct"] = tp
        else:
            log(f"config trailing_punct={data['trailing_punct']!r} 应为 auto/strip/keep, 已回退 auto")
    for k in ("uppercase_abbr", "latin_keep_case"):
        if k in data:
            v = data[k]
            if isinstance(v, str):
                CFG[k] = [v]
            elif isinstance(v, (list, tuple)):
                CFG[k] = [str(x) for x in v]
            else:
                log(f"配置 {k} 应该是字符串列表, 已忽略")
                CFG[k] = ["po"] if k == "uppercase_abbr" else []
    # 布尔项写成 "false"/0 时也能按字面意思理解
    for k in ("use_half", "enable_vad", "enable_punc", "paste", "restore_clipboard",
              "beep", "show_ui", "live_preview", "postprocess", "filler_removal",
              "number_norm", "diagnose", "fast_load", "mute_speaker"):
        if k in data and not isinstance(data[k], bool):
            v = data[k]
            if isinstance(v, str):
                CFG[k] = v.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(v, (int, float)):
                CFG[k] = bool(v)
    # 模型/VAD 内部硬编码 16kHz, 配置写别的值只会在识别时报错, 这里强制纠正
    if CFG.get("sample_rate") != 16000:
        log(f"config 的 sample_rate={CFG.get('sample_rate')!r} 无效: 模型固定 16kHz, 已强制回 16000")
        CFG["sample_rate"] = 16000


def _as_int(value, default, name):
    """配置值转整数, 非法时回退默认值并记日志(而不是静默崩溃)"""
    try:
        return int(float(str(value).strip()))
    except Exception:
        log(f"配置 {name}={value!r} 不是数字, 已改用默认值 {default}")
        return default


def _as_float(value, default, name):
    """配置值转浮点数, 非法时回退默认值并记日志"""
    try:
        return float(str(value).strip())
    except Exception:
        log(f"配置 {name}={value!r} 不是数字, 已改用默认值 {default}")
        return default


# ---- 剪贴板完整快照/还原: 不只存纯文本, 图片/文件/HTML 等全部格式都保留 ----
GMEM_MOVEABLE = 0x0002


class _METAFILEPICT(ctypes.Structure):
    _fields_ = [("mm", ctypes.c_long), ("xExt", ctypes.c_long),
                ("yExt", ctypes.c_long), ("hMF", ctypes.c_void_p)]


def _clip_api():
    """Win32 剪贴板函数必须显式声明 64 位返回类型, 否则句柄默认按 32 位 int 截断"""
    u, g, k = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.kernel32
    u.OpenClipboard.argtypes, u.OpenClipboard.restype = [ctypes.c_void_p], ctypes.c_int
    u.CloseClipboard.restype = ctypes.c_int
    u.EmptyClipboard.restype = ctypes.c_int
    u.EnumClipboardFormats.argtypes, u.EnumClipboardFormats.restype = [ctypes.c_uint], ctypes.c_uint
    u.GetClipboardData.argtypes, u.GetClipboardData.restype = [ctypes.c_uint], ctypes.c_void_p
    u.SetClipboardData.argtypes, u.SetClipboardData.restype = [ctypes.c_uint, ctypes.c_void_p], ctypes.c_void_p
    u.GetClipboardSequenceNumber.restype = ctypes.c_uint
    u.CopyImage.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    u.CopyImage.restype = ctypes.c_void_p
    g.CopyEnhMetaFileW.argtypes, g.CopyEnhMetaFileW.restype = [ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p
    g.CopyMetaFileW.argtypes, g.CopyMetaFileW.restype = [ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p
    g.GetPaletteEntries.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    g.GetPaletteEntries.restype = ctypes.c_uint
    g.CreatePalette.argtypes, g.CreatePalette.restype = [ctypes.c_void_p], ctypes.c_void_p
    k.GlobalAlloc.argtypes, k.GlobalAlloc.restype = [ctypes.c_uint, ctypes.c_size_t], ctypes.c_void_p
    k.GlobalLock.argtypes, k.GlobalLock.restype = [ctypes.c_void_p], ctypes.c_void_p
    k.GlobalUnlock.argtypes = [ctypes.c_void_p]
    k.GlobalSize.argtypes, k.GlobalSize.restype = [ctypes.c_void_p], ctypes.c_size_t
    k.GlobalFree.argtypes = [ctypes.c_void_p]
    u.RegisterClipboardFormatW.argtypes, u.RegisterClipboardFormatW.restype = [ctypes.c_wchar_p], ctypes.c_uint
    return u, g, k


def _clip_open(u):
    for _ in range(10):        # 剪贴板常被其它程序短暂占用, 重试约 0.2s
        if u.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def clipboard_sequence():
    """剪贴板序号: 内容每被改动一次就 +1, 用于还原前判断期间是否有人动过"""
    try:
        u = ctypes.windll.user32
        u.GetClipboardSequenceNumber.restype = ctypes.c_uint
        return int(u.GetClipboardSequenceNumber())
    except Exception:
        return -1


def clipboard_snapshot():
    """把剪贴板里所有格式的内容复制进内存。
    返回 [(格式, 数据, 是否GDI句柄), ...] (空列表 = 剪贴板本来就是空的);
    快照失败返回 None (此时不能还原, 否则会把剪贴板弄丢)。
    图片(CF_BITMAP/DIB/PNG)、文件(CF_HDROP)、HTML/RTF 等富格式都能完整保留。"""
    items = []
    try:
        u, g, k = _clip_api()
        if not _clip_open(u):
            return None
        try:
            fmt, seen = 0, set()
            while True:
                fmt = u.EnumClipboardFormats(fmt)
                if not fmt or fmt in seen:
                    break
                seen.add(fmt)
                h = u.GetClipboardData(fmt)
                if not h:
                    continue      # owner-display 等拿不到数据, 还原不了, 跳过
                if fmt == 2:      # CF_BITMAP: GDI 句柄, 不能按内存块读
                    hc = u.CopyImage(h, 0, 0, 0, 0)
                    if hc:
                        items.append((fmt, hc, True))
                elif fmt == 14:   # CF_ENHMETAFILE
                    hc = g.CopyEnhMetaFileW(h, None)
                    if hc:
                        items.append((fmt, hc, True))
                elif fmt == 9:    # CF_PALETTE
                    n = g.GetPaletteEntries(h, 0, 0, None)
                    if n:
                        buf = (ctypes.c_ubyte * (4 + 4 * n))()   # LOGPALETTE
                        ctypes.memmove(buf, (ctypes.c_ushort * 2)(0x0300, n), 4)
                        g.GetPaletteEntries(h, 0, n, ctypes.byref(buf, 4))
                        hp = g.CreatePalette(ctypes.cast(buf, ctypes.c_void_p))
                        if hp:
                            items.append((fmt, hp, True))
                elif fmt == 3:    # CF_METAFILEPICT: 结构体里套着图元文件句柄
                    size = k.GlobalSize(h)
                    p = k.GlobalLock(h)
                    if p and size and size >= ctypes.sizeof(_METAFILEPICT):
                        try:
                            m = _METAFILEPICT.from_buffer_copy(ctypes.string_at(p, size))
                            hm = g.CopyMetaFileW(m.hMF, None)
                            if hm:
                                m.hMF = hm
                                items.append((fmt, bytes(m), False))
                        finally:
                            k.GlobalUnlock(h)
                else:             # 普通格式(文本/文件列表/DIB/PNG/HTML...): 内存块整体复制
                    size = k.GlobalSize(h)
                    p = k.GlobalLock(h) if size else None
                    if p:
                        try:
                            items.append((fmt, ctypes.string_at(p, size), False))
                        finally:
                            k.GlobalUnlock(h)
            return items
        finally:
            u.CloseClipboard()
    except Exception:
        log("剪贴板快照失败:\n" + traceback.format_exc())
        return None


def clipboard_restore(items):
    """把快照原样写回剪贴板(先 EmptyClipboard 再逐格式 SetClipboardData)。
    返回 False = 剪贴板没打开/中途失败, 调用方可做兜底"""
    if items is None:
        return True
    try:
        u, g, k = _clip_api()
        if not _clip_open(u):
            return False
        try:
            u.EmptyClipboard()
            _set_clip_privacy_markers(u, k)   # 还原不算“新复制”, 别再记进历史
            for fmt, data, is_gdi in items:
                if is_gdi:        # data 是复制出来的 GDI 句柄, 交给系统接管
                    u.SetClipboardData(fmt, data)
                    continue
                h = k.GlobalAlloc(GMEM_MOVEABLE, len(data))
                if not h:
                    continue
                p = k.GlobalLock(h)
                if p:
                    ctypes.memmove(p, data, len(data))
                    k.GlobalUnlock(h)
                if not u.SetClipboardData(fmt, h):
                    k.GlobalFree(h)   # 没放进去就释放, 避免泄漏
            return True
        finally:
            u.CloseClipboard()
    except Exception:
        log("剪贴板还原失败:\n" + traceback.format_exc())
        return False


def _set_clip_privacy_markers(u, k):
    """打上微软定义的隐私标记格式: 本次写入不进 Win+V 剪贴板历史、不同步云剪贴板。
    (密码管理器同款机制; 标记只对本次写入生效, 不影响用户之后正常复制)"""
    for name in ("ExcludeClipboardContentFromMonitorProcessing",   # 排除一切剪贴板监视
                 "CanIncludeInClipboardHistory",                    # DWORD 0 = 不进历史
                 "CanUploadToCloudClipboard"):                       # DWORD 0 = 不上传云端
        fmt = u.RegisterClipboardFormatW(name)
        if not fmt:
            continue
        h = k.GlobalAlloc(GMEM_MOVEABLE, 4)
        if not h:
            continue
        p = k.GlobalLock(h)
        if p:
            ctypes.memmove(p, b"\x00\x00\x00\x00", 4)
            k.GlobalUnlock(h)
        if not u.SetClipboardData(fmt, h):
            k.GlobalFree(h)


def clipboard_copy_excluded(text):
    """把识别文本写上剪贴板并带隐私标记: 粘贴照常, 但 Win+V 历史里不会再堆满识别结果"""
    u, g, k = _clip_api()
    if not _clip_open(u):
        return False
    try:
        u.EmptyClipboard()
        _set_clip_privacy_markers(u, k)
        data = (text + "\x00").encode("utf-16-le")
        h = k.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not h:
            return False
        p = k.GlobalLock(h)
        if not p:
            k.GlobalFree(h)
            return False
        ctypes.memmove(p, data, len(data))
        k.GlobalUnlock(h)
        if not u.SetClipboardData(13, h):    # CF_UNICODETEXT
            k.GlobalFree(h)
            return False
        return True
    finally:
        u.CloseClipboard()


# ---- 逐字输入: 直接 SendInput(KEYEVENTF_UNICODE), 不受输入法/键盘布局影响 ----
_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", _ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", _ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004


def _type_text(text, delay=0.005):
    """逐字输入文本. 每个字符都走 Unicode 注入, 中文/英文/数字一致可靠.

    以前用 pynput 的 Controller.type, ASCII 走虚拟键码, 会被输入法/键盘布局干扰,
    实测会丢字(例如 "abc123" 变成 "abc23"), 所以改成统一的 Unicode 方式.
    """
    for ch in text:
        if ch == "\r":
            continue
        if ch == "\n":                       # 换行用回车键
            for flags in (0, _KEYEVENTF_KEYUP):
                ev = _INPUT(type=1, u=_INPUT._U(
                    ki=_KEYBDINPUT(0x0D, 0x1C, flags, 0, 0)))
                ctypes.windll.user32.SendInput(1, ctypes.byref(ev), ctypes.sizeof(_INPUT))
            time.sleep(delay)
            continue
        units = [ch]
        if ord(ch) > 0xFFFF:                 # 非 BMP 字符拆成 UTF-16 代理对
            v = ord(ch) - 0x10000
            units = [chr(0xD800 + (v >> 10)), chr(0xDC00 + (v & 0x3FF))]
        for u in units:
            code = ord(u)
            down = _INPUT(type=1, u=_INPUT._U(
                ki=_KEYBDINPUT(0, code, _KEYEVENTF_UNICODE, 0, 0)))
            up = _INPUT(type=1, u=_INPUT._U(
                ki=_KEYBDINPUT(0, code, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP, 0, 0)))
            arr = (_INPUT * 2)(down, up)
            ctypes.windll.user32.SendInput(2, arr, ctypes.sizeof(_INPUT))
            time.sleep(delay)


def parse_key(name):
    """把 'f9' / 'scroll_lock' / 'a' / 'none' 转成 pynput 按键对象"""
    if not name or str(name).lower() == "none":
        return None
    name = str(name)
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    return name


def _prefer_fp16():
    """RTX 20 系及更老 (SM<8.0) 没有原生 bf16, 但 PyTorch 仍报"支持",
    库会选 bfloat16 — 实测比 fp16 慢一倍. 这里把它掰回 fp16."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability(0) >= (8, 0):
            return False
        if not torch.cuda.is_bf16_supported():
            return False
        torch.cuda.is_bf16_supported = lambda: False
        return True
    except Exception:
        return False


def _patch_aed_fast_load():
    """ASR 权重改成 fp16 + mmap 加载(需要先跑 make_fp16.py).

    原来的 fp32 checkpoint 有 4.4GB, 加载时 torch.load 复制一份、load_state_dict
    再复制一份, 峰值接近 9GB, 之后那 4.4GB 也未必还给系统(常驻 4~5GB).
    fp16 文件只有 2.2GB, 配合 mmap + assign 直接把权重映射进模型, 峰值降到 2GB 左右.
    没有 fp16 文件时自动回退到原来的加载方式.
    """
    try:
        import torch
        import fireredasr2s.fireredasr2.asr as asr_mod
        from fireredasr2s.fireredasr2.models.fireredasr_aed import FireRedAsrAed
    except Exception as e:
        return "无法载入 fireredasr2 (%s)" % e
    p = os.path.join(MODELS, "FireRedASR2-AED", "model.fp16.pth.tar")
    fp32 = os.path.join(MODELS, "FireRedASR2-AED", "model.pth.tar")
    if not os.path.exists(p):
        return "原 fp32 (没有 model.fp16.pth.tar)"
    if not (CFG.get("fast_load", True) and CFG.get("use_half", True)):
        if os.path.exists(fp32):
            return "原 fp32"
        # fp32 原文件已删掉, 只剩 fp16 权重, 半精度跟着强制打开
        CFG["use_half"] = True

    orig = asr_mod.load_fireredasr_aed_model

    def fast_load(model_path):
        fp16_path = model_path[:-len("model.pth.tar")] + "model.fp16.pth.tar"
        if not os.path.exists(fp16_path):
            return orig(model_path)
        try:
            ck = torch.load(fp16_path, map_location="cpu", mmap=True, weights_only=False)
            old = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float16)
                # 直接在显卡上建模: 否则会先在内存里分配 2.2GB fp16 权重,
                # 这些内存还给系统很慢(实测常驻 2.8GB), 建在显卡上则内存几乎不涨
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    with torch.device(dev):
                        model = FireRedAsrAed.from_args(ck["args"])
                except Exception:
                    model = FireRedAsrAed.from_args(ck["args"])
            finally:
                torch.set_default_dtype(old)
            model.load_state_dict(ck["model_state_dict"], strict=False, assign=True)
            model = model.to(torch.float16)
            return model.cuda() if torch.cuda.is_available() else model
        except Exception:
            log("fp16 快速加载失败, 回退到 fp32:\n" + traceback.format_exc())
            return orig(model_path)

    asr_mod.load_fireredasr_aed_model = fast_load
    return "fp16+mmap (%.1f GB)" % (os.path.getsize(p) / 1024 ** 3)


class SpeakerMute:
    """录音期间静音默认扬声器, 结束后恢复 (config: mute_speaker)

    防幻听: 电脑正在放的音乐/视频/提示音会被麦克风收进去, 识别成幽灵文字。
    pycaw 未安装或非 Windows 时自动降级为空操作; 任何失败都只记日志, 不影响录音。
    每次都在当前线程重新拿 COM 接口 (start 在热键线程, stop 可能在看门狗线程,
    COM 接口不跨线程复用); 只记得"进入前是否已静音", 不替用户取消他自己设的静音。
    限制: 只管系统默认播放设备; 进程被强杀(taskkill /F)时来不及恢复, 需手动取消静音。
    """

    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.active = False       # 当前是否处于"由本程序静音"的状态
        self._was_muted = False   # 进入前的原状态
        self._warned = False      # pycaw 缺失只提醒一次

    def _endpoint(self):
        """在当前线程初始化 COM 并拿默认播放设备的音量接口, 拿不到返回 None"""
        if os.name != "nt":
            return None
        try:
            import comtypes
            try:                  # 工作线程可能还没初始化 COM; 已初始化时调用无害
                comtypes.CoInitialize()
            except OSError:
                pass
            from pycaw.pycaw import AudioUtilities
            return AudioUtilities.GetSpeakers().EndpointVolume
        except Exception:
            return None

    def start(self):
        """录音开始: 记住原状态并静音. 任何失败都只降级, 绝不抛异常"""
        if not self.enabled or self.active:
            return
        vol = self._endpoint()
        if vol is None:
            if not self._warned:
                self._warned = True
                log("mute_speaker 需要 pycaw: pip install pycaw comtypes (仅提示一次)")
            return
        try:
            self._was_muted = bool(vol.GetMute())
            if not self._was_muted:
                vol.SetMute(1, None)
                self.active = True
                log("录音期间已静音扬声器(防幻听), 结束后恢复")
            # 进入前本来就是静音的就不动它, 结束时也无需恢复
        except Exception:
            log("扬声器静音失败(不影响录音):\n" + traceback.format_exc())

    def stop(self):
        """录音结束: 恢复进入前的状态. 重复调用/未静音时是空操作"""
        if not self.active:
            return
        self.active = False
        if self._was_muted:
            return                # 进入前本来就是静音的, 不替用户取消静音
        try:
            vol = self._endpoint()
            if vol is None:
                log("扬声器恢复失败(拿不到音频设备), 可能需要手动取消静音")
                return
            vol.SetMute(0, None)
            log("扬声器已恢复")
        except Exception:
            log("扬声器恢复失败, 可能需要手动取消静音:\n" + traceback.format_exc())


class Overlay:
    """屏幕底部浮窗: 录音状态 / 实时草稿 / 最终结果"""

    W, H = 380, 96
    BG = "#12171c"
    GREEN = "#31c48d"
    AMBER = "#f0a020"
    BLUE = "#4aa3ff"
    GRAY = "#8894a0"

    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.state = "idle"
        self.t_rec = 0.0
        self.level = 0.0
        self.level_max = 0.0
        self.hide_at = 0.0
        self.hwnd = None
        self.mute = None         # SpeakerMute 对象(由 VoiceInput.run 注入); 非空且 active 时标题显示"静音外放"

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.0)
        except Exception:
            pass
        root.configure(bg=self.BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.x = (sw - self.W) // 2
        self.y = sh - self.H - 120
        # 平时放在屏幕外 + 全透明, 显示时只移动位置, 不重新映射窗口
        # (重新映射会抢前台, 导致重启后第一次输入跑到浮窗上)
        self.offx, self.offy = sw + 600, sh + 600
        root.geometry("%dx%d+%d+%d" % (self.W, self.H, self.offx, self.offy))

        self.cv = tk.Canvas(root, width=self.W, height=self.H, bg=self.BG,
                            highlightthickness=0)
        self.cv.pack()
        self.dot = self.cv.create_oval(16, 16, 26, 26, fill=self.GREEN, outline="")
        self.title = self.cv.create_text(36, 21, anchor="w", text="",
                                         fill="#ffffff",
                                         font=("Microsoft YaHei UI", 11, "bold"))
        self.body = self.cv.create_text(16, 42, anchor="nw", text="",
                                        fill="#cfd8e3", width=self.W - 32,
                                        font=("Microsoft YaHei UI", 10))
        self.bar_bg = self.cv.create_rectangle(16, 86, self.W - 16, 90,
                                               fill="#242c33", outline="")
        self.bar = self.cv.create_rectangle(16, 86, 16, 90, fill=self.GREEN,
                                            outline="")
        root.update_idletasks()
        self._no_activate()
        root.after(80, self._poll)

    # ---- win32: 不抢焦点 + 鼠标穿透 ----
    def _no_activate(self):
        try:
            user32 = ctypes.windll.user32
            hwnd = self.root.winfo_id()
            parent = user32.GetParent(hwnd)
            if parent:
                hwnd = parent
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                  style | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
            self.hwnd = hwnd
        except Exception:
            self.hwnd = None

    def _topmost(self):
        try:
            self.root.attributes("-topmost", True)
            if self.hwnd:
                # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
                ctypes.windll.user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0,
                                                  0x0001 | 0x0002 | 0x0010)
        except Exception:
            pass

    def _show(self):
        # 注意顺序: Tk 在 Windows 上执行 attributes("-topmost") 会把窗口位置
        # 弹回上一次记录的坐标, 所以必须"先置顶, 后定位"。
        try:
            self.root.attributes("-alpha", 0.9)
        except Exception:
            pass
        self._topmost()
        self.root.geometry("%dx%d+%d+%d" % (self.W, self.H, self.x, self.y))

    def _hide(self):
        try:
            self.root.attributes("-alpha", 0.0)
        except Exception:
            pass
        self.root.geometry("%dx%d+%d+%d" % (self.W, self.H, self.offx, self.offy))
        self.state = "idle"

    # ---- 线程安全的对外接口 ----
    def set_state(self, state, text="", level=None):
        self.q.put((state, text, level))

    def set_level(self, level):
        self.level = level

    def _apply(self, state, text):
        self.state = state
        if state == "rec":
            self.t_rec = time.time()
            self.level_max = 0.0
            self.cv.itemconfigure(self.dot, fill=self.GREEN)
            self.cv.itemconfigure(self.body, text="", fill="#cfd8e3")
            self.cv.itemconfigure(self.bar_bg, state="normal")
            self.cv.itemconfigure(self.bar, state="normal")
            self._show()
        elif state == "draft":
            self.cv.itemconfigure(self.dot, fill=self.GREEN)
            self.cv.itemconfigure(self.body, text=(text or "")[-100:], fill="#cfd8e3")
            self.cv.itemconfigure(self.bar_bg, state="normal")
            self.cv.itemconfigure(self.bar, state="normal")
            self._show()
        elif state == "proc":
            self.cv.itemconfigure(self.dot, fill=self.AMBER)
            self.cv.itemconfigure(self.title, text="识别中…")
            self.cv.itemconfigure(self.body, text=(text or "")[-100:], fill="#cfd8e3")
            self.cv.itemconfigure(self.bar_bg, state="hidden")
            self.cv.itemconfigure(self.bar, state="hidden")
            self._show()
        elif state == "ok":
            self.cv.itemconfigure(self.dot, fill=self.BLUE)
            self.cv.itemconfigure(self.title, text="✓ 已输入")
            self.cv.itemconfigure(self.body, text=(text or "")[-100:], fill="#dfe8f3")
            self.cv.itemconfigure(self.bar_bg, state="hidden")
            self.cv.itemconfigure(self.bar, state="hidden")
            self._show()
            self.hide_at = time.time() + 2.2
        else:  # info
            self.cv.itemconfigure(self.dot, fill=self.GRAY)
            self.cv.itemconfigure(self.title, text=text or "")
            self.cv.itemconfigure(self.body, text="", fill="#8894a0")
            self.cv.itemconfigure(self.bar_bg, state="hidden")
            self.cv.itemconfigure(self.bar, state="hidden")
            self._show()
            self.hide_at = time.time() + 1.8

    def _poll(self):
        try:
            while True:
                state, text, level = self.q.get_nowait()
                if level is not None:
                    self.level = level
                self._apply(state, text)
        except queue.Empty:
            pass

        now = time.time()
        if self.state in ("rec", "draft"):
            t = now - self.t_rec
            badge = " · 静音外放" if self.mute is not None and self.mute.active else ""
            self.cv.itemconfigure(self.title, text="● 录音中  %.1fs%s" % (t, badge))
            self.level_max = max(self.level_max, self.level)
            frac = min(1.0, (self.level / 4000.0) ** 0.5)
            self.cv.coords(self.bar, 16, 86, 16 + frac * (self.W - 32), 90)
            if t > 1.2 and self.level_max < 25:
                self.cv.itemconfigure(self.body, text="没检测到声音，检查麦克风", fill="#f0a020")
        elif self.state in ("ok", "info") and now > self.hide_at:
            self._hide()

        self.root.after(80, self._poll)


def join_sentences(res, gap_ms=2500):
    """把 VAD 切出来的相邻短片断按停顿长短合并: 停顿短 -> 逗号, 停顿长 -> 保留句号"""
    sents = res.get("sentences") or []
    if len(sents) < 2 or gap_ms <= 0:
        return None
    parts = []
    for i, s in enumerate(sents):
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if i < len(sents) - 1:
            nxt = (sents[i + 1].get("text") or "").lstrip()
            try:
                gap = int(sents[i + 1].get("start_ms", 0)) - int(s.get("end_ms", 0))
            except Exception:
                gap = gap_ms + 1
            # 英文句号后接大写字母, 认为本来就是新句子
            keep_period = t.endswith(".") and nxt[:1].isupper() and not t.endswith("..")
            if gap <= gap_ms and not keep_period:
                if t.endswith("。"):
                    t = t[:-1] + "，"
                elif t.endswith(".") and not t.endswith(".."):
                    t = t[:-1] + ", "
                elif not t.endswith(("，", ",", "！", "？", "!", "?", "；", ";")):
                    t = t + "，"
        parts.append(t)
    return "".join(parts) or None


class VoiceInput:
    def __init__(self, cfg):
        self.cfg = cfg
        self.recording = False
        self.stream = None
        self.chunks = []
        self.rec_lock = threading.Lock()   # 录音开始/结束互斥: 防止看门狗和热键同时停/开
        self.q = queue.Queue()
        self.kb = keyboard.Controller()
        self.system = None
        self.down = set()
        self.ui = None
        self.mute = SpeakerMute(cfg.get("mute_speaker"))
        atexit.register(self.mute.stop)    # 退出(含 Ctrl+C)时兜底恢复扬声器
        self.model_lock = threading.Lock()
        # 剪贴板还原状态: 连续两次识别时, 后一次不能把前一次识别出的文字当成"原剪贴板"
        self._clip_lock = threading.Lock()
        self._clip_pending = False
        self._clip_original = None
        self._clip_gen = 0
        self.rules = postprocess.load_rules(os.path.join(BASE, "replacements.txt"), log)
        if self.rules:
            log(f"替换规则已加载 {len(self.rules)} 条: " +
                ", ".join(f"{a}=>{b}" for a, b in self.rules[:8]))
        self.hotkey = parse_key(cfg.get("hotkey", "f7"))
        self.toggle_key = parse_key(cfg.get("toggle_key", "f10"))
        for label, raw, key in (("长按说话键", cfg.get("hotkey"), self.hotkey),
                                ("开关热键", cfg.get("toggle_key"), self.toggle_key)):
            if isinstance(key, str):
                log(f"警告: {label} {raw!r} 不是有效的按键名, 这个键不会生效; "
                    f"请双击「设置热键.bat」重新选择")

    # ---------- model ----------
    def load_model(self):
        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredvad import FireRedVadConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig

        t0 = time.time()
        if _prefer_fp16():
            log("显卡没有原生 bf16 (SM<8.0), 已改用 FP16 推理(快一倍)")
        log("ASR 权重加载方式: " + _patch_aed_fast_load())
        # VAD 参数单位是帧(10ms/帧)。默认 0.6s 停顿才算断句, 1.4s 内的间隔合并成一段,
        # 避免说话中间稍微停顿就被拆成好几个短句。
        def _frames(ms, default):
            try:
                return max(1, int(float(ms) / 10))
            except Exception:
                return default

        vad_cfg = FireRedVadConfig(
            use_gpu=False,
            min_silence_frame=_frames(self.cfg.get("vad_min_silence_ms"), 60),
            merge_silence_frame=_frames(self.cfg.get("vad_merge_silence_ms"), 140),
            max_speech_frame=_frames(self.cfg.get("vad_max_speech_ms"), 2000),
        )
        c = FireRedAsr2SystemConfig(
            vad_model_dir=os.path.join(MODELS, "FireRedVAD", "VAD"),
            lid_model_dir=os.path.join(MODELS, "FireRedLID"),
            asr_model_dir=os.path.join(MODELS, "FireRedASR2-AED"),
            punc_model_dir=os.path.join(MODELS, "FireRedPunc"),
            asr_type="aed",
            enable_vad=self.cfg["enable_vad"],
            enable_lid=False,
            enable_punc=self.cfg["enable_punc"],
            # 长语音被 VAD 切成多段后, 多段一起送 GPU 批量推理(库默认串行逐段)
            asr_batch_size=8,
            punc_batch_size=8,
            asr_config=FireRedAsr2Config(use_gpu=True, use_half=self.cfg["use_half"],
                                         beam_size=_as_int(self.cfg.get("beam_size", 3),
                                                           3, "beam_size"),
                                         return_timestamp=False),
            vad_config=vad_cfg,
            punc_config=FireRedPuncConfig(use_gpu=True),
        )
        self.system = FireRedAsr2System(c)
        log(f"模型加载完成 {time.time()-t0:.1f}s")

        # 热词(解码偏置 + 输出拼写校正)
        try:
            bonus = float(self.cfg.get("hotword_bonus", 2.0))
            if bonus > 0:
                hotword.install(self.system.asr.tokenizer,
                                os.path.join(BASE, "hotwords.txt"),
                                bonus=bonus, logger=log)
        except Exception:
            log("热词加载失败(不影响使用):\n" + traceback.format_exc())

        # 预热, 避免第一次识别特别慢
        try:
            warm = os.path.join(BASE, "assets", "hello_zh.wav")
            if not os.path.exists(warm):
                warm = (self.cfg["sample_rate"],
                        np.zeros(self.cfg["sample_rate"], dtype=np.int16))
            t1 = time.time()
            self.system.process(warm)
            log(f"预热完成 {time.time()-t1:.2f}s")
        except Exception:
            log("预热失败(不影响使用):\n" + traceback.format_exc())

    # ---------- audio ----------
    def start_rec(self):
        with self.rec_lock:
            if self.recording:
                return
            self.chunks = []
            self.level = 0.0
            sr = self.cfg["sample_rate"]

            def cb(indata, frames, t, status):
                if status:
                    log("audio status: " + str(status))
                self.chunks.append(indata.copy())
                if self.ui is None:
                    return          # 电平值只有浮窗用, 没开浮窗就别白算 RMS
                try:
                    a = indata.astype(np.float32)      # 只要个电平值, float32 足够
                    self.level = float(np.sqrt(np.mean(a * a)))
                    self.ui.set_level(self.level)
                except Exception:
                    pass

            stream = None
            self.mute.start()            # 先静音扬声器: 提示音/音乐不进麦克风
            try:
                stream = sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                        device=self.cfg["device"], callback=cb)
                stream.start()
            except Exception:
                if stream is not None:   # start 失败要把半开的流关掉, 否则占住麦克风
                    try:
                        stream.close()
                    except Exception:
                        pass
                self.mute.stop()         # 开流失败也要把扬声器恢复回来
                raise
            self.stream = stream
            self.recording = True
            self.t_rec = time.time()
            self.sid = self.t_rec
        if self.ui is not None:
            self.ui.set_state("rec")
        threading.Thread(target=self._watchdog, args=(self.sid,), daemon=True).start()
        if self.cfg.get("live_preview", False):
            threading.Thread(target=self._live_loop, args=(self.sid,), daemon=True).start()
        if self.cfg["beep"]:
            threading.Thread(target=_beep, args=(1100, 60), daemon=True).start()
        log("● 录音中 ...")

    def _live_loop(self, sid):
        """长句边说边出草稿: 定时把已录到的音频重新识别一遍, 只显示, 不输入"""
        sr = self.cfg["sample_rate"]
        interval = max(0.4, float(self.cfg.get("live_interval_ms", 1200)) / 1000.0)
        after = float(self.cfg.get("live_after_sec", 3.0))
        cap = int(float(self.cfg.get("live_max_sec", 10.0)) * sr)
        cost = 0.0
        while self.recording and getattr(self, "sid", None) == sid:
            time.sleep(max(interval, cost * 1.3))
            if not self.recording or getattr(self, "sid", None) != sid:
                return
            if time.time() - self.t_rec < after:
                continue
            if not self.model_lock.acquire(False):
                continue
            try:
                chunks = list(self.chunks)
                if not chunks:
                    continue
                audio = np.concatenate(chunks, axis=0).reshape(-1)
                if audio.size < sr:
                    continue
                cut = audio.size > cap
                if cut:
                    audio = audio[-cap:]
                t0 = time.time()
                res = self.system.process((sr, audio))
                cost = time.time() - t0
                text = (res.get("text") or "").strip()
                if text:
                    text = join_sentences(res, self.cfg.get("join_gap_ms", 2500)) or text
                    # 先规则改写(逐字母串转小写等), 再让热词把已知词的标准大小写改回来
                    text = postprocess.apply(text, self.cfg, self.rules)
                    text = hotword.fix_text(text)
                    if cut:
                        text = "…" + text
                    if self.ui is not None and self.recording:
                        self.ui.set_state("draft", text)
            except Exception:
                log("实时预览出错:\n" + traceback.format_exc())
            finally:
                self.model_lock.release()

    def _watchdog(self, sid):
        while self.recording and getattr(self, "sid", None) == sid:
            if time.time() - self.t_rec > self.cfg["max_sec"]:
                log(f"录音超过 {self.cfg['max_sec']}s, 自动停止")
                self.stop_rec()
                return
            time.sleep(0.5)

    def stop_rec(self):
        with self.rec_lock:
            if not self.recording:
                return
            self.recording = False       # 先置标志: watchdog 和热键释放可能同时调进来
            dur = time.time() - self.t_rec
            stream, self.stream = self.stream, None
            chunks = self.chunks          # 用局部副本处理, 新录音随时可以把 self.chunks 清零
            self.chunks = []
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    try:
                        stream.close()    # stop 报错也不能跳过 close, 否则设备被占用
                    except Exception:
                        pass
            self.mute.stop()              # 恢复扬声器(后面"说话太短"等提前 return 前也要恢复)
        if self.cfg["beep"]:
            threading.Thread(target=_beep, args=(780, 60), daemon=True).start()
        if dur < self.cfg["min_sec"] or not chunks:
            log(f"录音太短 ({dur:.2f}s), 忽略")
            if self.ui is not None:
                self.ui.set_state("info", "说话太短")
            return
        if self.ui is not None:
            self.ui.set_state("proc")
        # (N,1) 拼成 (N,1), 再展平成 (N,): 与原来“写 wav 再 sf.read 回来”的形状一致,
        # VAD/ASR 的特征提取都要求一维数组
        audio = np.concatenate(chunks, axis=0).reshape(-1)
        # 自动增益: 麦克风音量偏小时把电平拉到合适范围(最多放大 20 倍)
        f = audio.astype(np.float32)
        peak = float(np.abs(f).max()) if f.size else 0.0
        if peak > 30:
            gain = min(0.6 * 32767 / peak, 20.0)
            if gain > 1.05:
                f = np.clip(f * gain, -32768, 32767)
                log(f"自动增益 x{gain:.1f} (峰值 {peak:.0f})")
            audio = f.astype(np.int16)
        try:
            os.makedirs(os.path.dirname(TMP_WAV), exist_ok=True)   # logs 目录被删后也能自愈
        except Exception:
            pass
        log(f"○ 录音结束 {dur:.2f}s -> 排队识别")
        self.q.put(audio)     # int16 数组直接进队列, 全程内存, 不落盘

    # ---------- worker ----------
    def worker(self):
        while True:
            audio = self.q.get()      # int16 单声道数组, 从录音到识别全程不落盘
            try:
                sr = self.cfg["sample_rate"]
                diag = bool(self.cfg.get("diagnose"))
                dur = audio.shape[0] / sr
                if diag:
                    try:
                        peak = int(np.abs(audio).max()) if audio.size else 0
                        log(f"[诊断] 音频 {dur:.2f}s 峰值 {peak}")
                        sf.write(os.path.join(BASE, "logs", "last.wav"), audio, sr)
                    except Exception:
                        log("[诊断] 保存诊断音频失败:\n" + traceback.format_exc())
                t0 = time.time()
                with self.model_lock:      # 与实时草稿互斥, 避免同时占用显存
                    res = self.system.process((sr, audio))
                cost = time.time() - t0
                if diag:
                    try:
                        segs = res.get("vad_segments_ms") or []
                        log(f"[诊断] VAD {len(segs)} 段: {segs}")
                        for s in (res.get("sentences") or []):
                            log(f"[诊断]   {s.get('start_ms')}-{s.get('end_ms')}ms "
                                f"{(s.get('text') or '')!r}")
                    except Exception:
                        log("[诊断] 记录分段失败:\n" + traceback.format_exc())
                text = (res.get("text") or "").strip()
                joined = join_sentences(res, self.cfg.get("join_gap_ms", 2500))
                if joined:
                    text = joined
                if text:
                    # 先规则改写(逐字母串转小写等), 再让热词把已知词的标准大小写改回来
                    text = postprocess.apply(text, self.cfg, self.rules)
                    text = hotword.fix_text(text)
                if not text:
                    log(f"未识别到内容 ({cost:.2f}s)")
                    if self.ui is not None:
                        self.ui.set_state("info", "未识别到内容")
                    continue
                log(f"√ [{cost:.2f}s] {text}")
                if dur >= 2.0 and len(text) <= 3:
                    log(f"  ! 音频 {dur:.2f}s 却只识别出 {len(text)} 个字符, "
                        f"若反复出现请把 config.json 的 diagnose 设为 true 再复现一次")
                self.inject(text)
                if self.ui is not None:
                    self.ui.set_state("ok", text)
            except Exception:
                log("识别出错:\n" + traceback.format_exc())
                if self.ui is not None:
                    self.ui.set_state("info", "识别出错，见日志")

    # ---------- inject ----------
    def inject(self, text):
        if self.cfg.get("dry_run"):
            log("[dry-run] 不输入: " + text)
            return
        if self.cfg["paste"]:
            # 0.5 秒内连续识别两次时, 第二次读到的剪贴板其实是第一次的识别结果,
            # 直接还原会把用户原来的内容弄丢。这里共用"第一次的原剪贴板", 只还原最早的。
            with self._clip_lock:
                if self.cfg["restore_clipboard"] and not self._clip_pending:
                    self._clip_pending = True
                    snap = clipboard_snapshot()
                    if snap is None:
                        # 完整快照失败(剪贴板被长期占用), 退化到只存纯文本: 文本至少还能还回去
                        try:
                            t = pyperclip.paste()
                            snap = [(13, (t + "\x00").encode("utf-16-le"), False)] if t else []
                        except Exception:
                            snap = None
                    self._clip_original = snap
                self._clip_gen += 1
                gen = self._clip_gen
            if not clipboard_copy_excluded(text):
                pyperclip.copy(text)   # 兜底: 标记写不进去也要保证能粘贴
            seq = clipboard_sequence()
            time.sleep(0.06)
            with self.kb.pressed(keyboard.Key.ctrl):
                self.kb.press("v")
                self.kb.release("v")
            if self.cfg["restore_clipboard"]:
                def restore():
                    time.sleep(0.5)
                    with self._clip_lock:
                        if self._clip_gen != gen:   # 后面还有新的识别, 交给它还原
                            return
                        self._clip_pending = False
                        data = self._clip_original
                        self._clip_original = None
                    if data is None:
                        return          # 快照失败, 宁可不动剪贴板也别把它弄丢
                    try:
                        if clipboard_sequence() != seq:
                            return      # 粘贴后用户又复制了新内容, 还原会把它覆盖掉
                        if not clipboard_restore(data):
                            # 剪贴板被长期锁死(极少见), 至少把原文本写回
                            try:
                                for f, d, gdi in data:
                                    if f == 13 and not gdi:
                                        pyperclip.copy(
                                            d.decode("utf-16-le", "ignore").rstrip("\x00"))
                                        break
                            except Exception:
                                pass
                    except Exception:
                        pass
                threading.Thread(target=restore, daemon=True).start()
        else:
            _type_text(text)

    # ---------- hotkey ----------
    def on_press(self, key):
        if key in self.down:      # 忽略系统按键重复
            return
        self.down.add(key)
        try:
            if self.toggle_key is not None and key == self.toggle_key:
                if self.recording:
                    self.stop_rec()
                else:
                    self.start_rec()
            elif self.hotkey is not None and key == self.hotkey and not self.recording:
                self.start_rec()
        except Exception as e:
            if isinstance(e, sd.PortAudioError):
                log(f"麦克风打不开({e}); 请检查 config.json 的 device, "
                    f"或用 --list-devices 查看可用设备")
            else:
                log("启动录音出错:\n" + traceback.format_exc())

    def on_release(self, key):
        self.down.discard(key)
        if self.hotkey is not None and key == self.hotkey:
            try:
                self.stop_rec()
            except Exception:
                log("停止录音出错:\n" + traceback.format_exc())

    def run(self):
        if self.cfg.get("postprocess", True):
            log("后处理: 开, 拼音模块(pypinyin): " +
                ("可用" if postprocess.lazy_pinyin else "未安装(中文域名不转拼音, 其它功能不受影响)"))
        self.load_model()
        threading.Thread(target=self.worker, daemon=True).start()

        root = None
        if self.cfg.get("show_ui", True):
            try:
                user32 = ctypes.windll.user32
                prev_fg = user32.GetForegroundWindow()
                root = tk.Tk()
                self.ui = Overlay(root)
                self.ui.mute = self.mute if self.mute.enabled else None
                log("浮窗已启用(屏幕底部); 关闭浮窗可在 config.json 设 \"show_ui\": false")
                # 双保险: 如果建窗口时不小心抢了前台, 立刻还给原窗口
                cur = user32.GetForegroundWindow()
                if prev_fg and cur != prev_fg and cur == self.ui.hwnd:
                    user32.SetForegroundWindow(prev_fg)
                    log("浮窗创建时抢了前台, 已还给原窗口")
            except Exception:
                self.ui = None
                root = None
                log("浮窗启动失败, 继续无界面运行:\n" + traceback.format_exc())

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        log(f"就绪: 按住 [{self.cfg.get('hotkey')}] 说话, 松开即输入; "
            f"[{self.cfg.get('toggle_key')}] 按一下开始/再按结束. Ctrl+C 退出")

        try:
            if root is not None:
                root.mainloop()      # Tk 必须占主线程, 键盘监听在后台线程
            else:
                listener.join()
        finally:
            self.mute.stop()         # 无论怎么退出都把扬声器恢复回来


def _beep(freq, ms):
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        pass


def _single_instance():
    """同一时间只允许跑一个实例(开机自启+手动启动不会重复)"""
    import ctypes
    h = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\VoiceTyping")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return h


def _prepare_audio(path):
    """--file 用: 校验音频并转成 16kHz 单声道, 返回 (路径, 是否临时文件)"""
    if not os.path.exists(path):
        raise RuntimeError("找不到音频文件: " + path)
    try:
        info = sf.info(path)
    except Exception as e:
        raise RuntimeError(f"无法读取音频文件 {path}: {e}")
    target_sr = _as_int(CFG.get("sample_rate", 16000), 16000, "sample_rate")
    if info.samplerate == target_sr and info.channels == 1:
        if info.frames < 800:      # 少于 0.05 秒: 模型内部会因空时间维崩溃
            raise RuntimeError(f"音频太短({info.frames} 帧, 至少需要 0.05 秒): {path}")
        return path, False
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"无法读取音频文件 {path}: {e}")
    if info.channels > 1:
        log(f"音频是 {info.channels} 声道, 已混成单声道")
    mono = data.mean(axis=1)
    if sr != target_sr:
        if mono.size < 2:
            raise RuntimeError("音频太短, 无法重采样")
        n = max(1, int(round(mono.size * target_sr / sr)))
        mono = np.interp(np.linspace(0, mono.size - 1, n),
                         np.arange(mono.size), mono).astype(np.float32)
        log(f"音频是 {sr} Hz, 已重采样到 {target_sr} Hz")
    if mono.size < 800:
        raise RuntimeError(f"音频太短({mono.size} 帧, 至少需要 0.05 秒): {path}")
    os.makedirs(TMP_WAV, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=TMP_WAV)
    os.close(fd)
    sf.write(tmp, mono, target_sr, subtype="PCM_16")
    return tmp, True


def transcribe_file(path):
    """--file: 转写一个音频文件后退出, 返回退出码"""
    try:
        src, tmp = _prepare_audio(path)
    except RuntimeError as e:
        log(str(e))
        return 2
    try:
        v = VoiceInput(CFG)
        v.load_model()
        res = v.system.process(src)
    except Exception:
        log("转写失败:\n" + traceback.format_exc())
        return 1
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass
    raw = (res.get("text") or "").strip()
    joined = join_sentences(res, CFG.get("join_gap_ms", 2500)) or raw
    text = postprocess.apply(joined, CFG, v.rules) if joined else ""
    text = hotword.fix_text(text) if text else ""
    print("RAW :", raw)
    print("POST:", text)
    return 0


def _cleanup_tmp():
    """清掉上次进程残留的临时录音/草稿(进程被强杀时来不及删), 免得 logs 慢慢变大。
    只动 60 秒前的文件, 避免误删其它实例正在用的; last.wav(诊断样本)和日志不动"""
    for d in dict.fromkeys([os.path.dirname(TMP_WAV), TMP_WAV]):
        try:
            names = os.listdir(d)
        except Exception:
            continue
        for f in names:
            if f == "last.wav" or not f.endswith(".wav"):
                continue
            if not (f.startswith("uttr_") or f.startswith("tmp") or f.startswith("_")):
                continue
            p = os.path.join(d, f)
            try:
                if time.time() - os.path.getmtime(p) > 60:
                    os.remove(p)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--type", action="store_true", help="逐字输入而不是粘贴")
    ap.add_argument("--no-punc", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--file", help="转写指定音频文件后退出(不常驻)")
    ap.add_argument("--dry-run", action="store_true", help="只识别不输入(测试用)")
    ap.add_argument("--no-post", action="store_true", help="关闭输出后处理(保留原始识别文本)")
    a = ap.parse_args()

    if a.list_devices:
        print(sd.query_devices())
        return

    # 先读 config.json, 再应用命令行参数, 否则参数会被配置文件里的值覆盖
    load_config()
    if a.device is not None:
        CFG["device"] = a.device
    if a.type:
        CFG["paste"] = False
    if a.no_punc:
        CFG["enable_punc"] = False
    if a.fp32:
        CFG["use_half"] = False
    if a.no_post:
        CFG["postprocess"] = False
    if a.dry_run:
        CFG["dry_run"] = True

    if a.file:
        return transcribe_file(a.file)

    log("=" * 50)
    log(f"启动 voice_typing, device={CFG['device']}, half={CFG['use_half']}, "
        f"paste={CFG['paste']}, beam={CFG.get('beam_size', 3)}, "
        f"post={CFG.get('postprocess', True)}")
    keep = _single_instance()
    if keep is None and not a.dry_run:
        log("已有实例在运行, 退出")
        return
    if keep is not None:
        # 只有确认自己是唯一实例才清理; dry-run 与正式实例共存时不 Cleanup,
        # 免得误删对方排队中的录音
        _cleanup_tmp()
    try:
        VoiceInput(CFG).run()
    except KeyboardInterrupt:
        log("退出")


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        log("退出")
        rc = 0
    except SystemExit:
        raise
    except Exception:
        # pythonw 没有控制台, 不写日志的话崩溃会完全静默
        log("程序异常退出:\n" + traceback.format_exc())
        rc = 1
    sys.exit(rc or 0)
