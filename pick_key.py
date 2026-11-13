# -*- coding: utf-8 -*-
"""按键拾取器: 按下你想用的键, 写入 config.json"""
import json
import os
import sys

from pynput import keyboard

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(BASE, "config.json")

HINTS = {
    "ctrl_r": "右Ctrl(推荐, 不影响输入, 所有键盘都有)",
    "f1": "帮助(多数软件)", "f3": "搜索", "f5": "刷新", "f6": "地址栏",
    "f9": "Typora 打字机模式", "f10": "激活菜单栏", "f11": "全屏",
    "f12": "开发者工具", "space": "空格(会误触)", "esc": "取消",
}


def key_name(key):
    if getattr(key, "name", None):
        return key.name
    if getattr(key, "char", None):
        return key.char
    return None


# 单独的左侧/中性修饰键容易和快捷键冲突, 默认跳过; 右侧 Ctrl 可用作说话键
SKIP = {"ctrl", "ctrl_l", "shift", "shift_l", "shift_r",
        "alt", "alt_l", "alt_r", "alt_gr", "cmd", "cmd_r", "caps_lock"}

# 只接受这些键, 避免把普通字母/数字设成热键(否则打字时会疯狂触发录音)
ALLOWED = {f"f{i}" for i in range(1, 25)} | {
    "scroll_lock", "pause", "print_screen", "insert", "menu", "ctrl_r",
}


def main():
    target = "toggle_key" if "--toggle" in sys.argv else "hotkey"
    label = "长按说话键" if target == "hotkey" else "切换录音键"

    print("=" * 56)
    print(f" 设置{label}")
    print(" 请按下一个你想要的按键 (按 Esc 取消)")
    print(" 建议: 右Ctrl / F7 / Pause 这类平时不用的键")
    print("=" * 56)

    got = {}

    def on_press(key):
        n = key_name(key)
        if not n:
            return
        if n == "esc":
            got["cancel"] = True
            return False
        if n in SKIP:
            if not got.get("_warned"):
                got["_warned"] = True
                print(f"  ({n} 是单独的修饰键, 容易和快捷键冲突, 请换一个键)")
            return
        if n not in ALLOWED:
            print(f"  ({n} 不适合做热键, 请用 F1~F24 / ScrollLock / Pause / PrintScreen)")
            return
        got["key"] = n
        return False

    with keyboard.Listener(on_press=on_press) as l:
        l.join()

    if got.get("cancel") or "key" not in got:
        print("已取消, 未做修改。")
        return 1

    name = got["key"]
    cfg = {}
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, "r", encoding="utf-8-sig") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    cfg[target] = name
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {label} = {name}")
    if name in HINTS:
        print(f"提示: {name} 在部分软件里是「{HINTS[name]}」, 若冲突请重选。")
    print("配置文件:", CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
