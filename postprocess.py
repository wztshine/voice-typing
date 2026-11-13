# -*- coding: utf-8 -*-
"""输出后处理: 纯规则(正则 + 查表), 在文字上屏前改写.

设计原则: 宁可不改, 也不改错.
- 只处理"形状稳定"的东西(数字+单位、句首语气词、被空格拆散的字母、固定缩写)
- 不在词表/规则里的内容一律不动
- 每条规则独立 try/except, 单条出错不影响其它规则
"""
import os
import re

try:
    from pypinyin import lazy_pinyin   # 中文域名转拼音: 百度 -> baidu
except Exception:
    lazy_pinyin = None

# ---------------- 数字 ----------------

CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
ALL_CN = "".join(CN_DIGIT.keys()) + "".join(CN_UNIT.keys())


def cn2int(s):
    """中文数字 -> int. 支持 十二 / 二十三 / 三百 / 一万 / 二零二六(逐位) / 一万二(=12000).
    失败返回 None"""
    if not s:
        return None
    # 逐位读法: 二零二六 -> 2026
    if len(s) > 1 and all(c in CN_DIGIT for c in s):
        return int("".join(str(CN_DIGIT[c]) for c in s))
    # 口语省略式: 万/千/百 有系数且后面只跟一个光数字时, 在数字后补上一级小单位
    # 一万二 -> 一万二千(12000), 四千五 -> 四千五百(4500), 一百五 -> 一百五十(150)
    # 系数必须有(万一/百五这种孤字不碰); 中间隔着零的不算(三万零五 = 30005);
    # 十后面跟数字是正常读法(三百二十一 = 321), 不展开
    m = RE_TRUNC.match(s)
    if m:
        s = m.group(1) + m.group(2) + {"万": "千", "千": "百", "百": "十"}[m.group(1)[-1]]
    total = section = num = 0
    for ch in s:
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            u = CN_UNIT[ch]
            if u == 10000:
                total += (section + num) * u
                section = num = 0
            else:
                section += (num or 1) * u
                num = 0
        else:
            return None
    return total + section + num


def _sub_num(m):
    v = cn2int(m.group("num"))
    return (str(v) + m.group("unit")) if v is not None else m.group(0)


def _sub_percent(m):
    body = m.group(1)
    if "点" in body:
        a, _, b = body.partition("点")
        x, y = cn2int(a), cn2int(b)
        if x is None or y is None:
            return m.group(0)
        return f"{x}.{y}%"
    v = cn2int(body)
    return m.group(0) if v is None else f"{v}%"


def _sub_time(m):
    h = cn2int(m.group("h"))
    rest = m.group("rest")
    if h is None:
        return m.group(0)
    if rest == "半":
        return f"{h}点{rest}"
    mm = re.match(r"([%s]{1,3})\s*分" % ALL_CN, rest)
    if not mm:
        return m.group(0)
    mv = cn2int(mm.group(1))
    return m.group(0) if mv is None else f"{h}点{mv}分"


def _sub_period_time(m):
    """下午三点五十分 -> 下午3点50分 (带不带分钟都转, 否则小时转了分钟还是汉字)"""
    h = cn2int(m.group("h"))
    if h is None:
        return m.group(0)
    out = m.group("per") + str(h) + "点"
    rest = m.group("rest")
    if not rest:
        return out
    if rest == "半":
        return out + "半"
    mm = re.match(r"([%s]{1,3})\s*分" % ALL_CN, rest)
    mv = cn2int(mm.group(1)) if mm else None
    return out + f"{mv}分" if mv is not None else m.group(0)


# ---------------- 规则表 ----------------
# 顺序有讲究: 先合并被拆散的英文, 再处理缩写/数字, 最后语气词与标点

RE_SPACED = re.compile(r"(?<![A-Za-z])([A-Za-z](?:[ \t]+[A-Za-z]){2,})(?![A-Za-z])")
RE_DOTCOM = re.compile(r"(?<=[A-Za-z])\s+dot\s+com(?![A-Za-z])", re.I)
# 中文"点"当域名点用: www点baidu点com -> www.baidu.com
# 只在"字母 + 点 + 字母"时替换, 中文里的"点"不受影响
RE_CN_DOT = re.compile(r"(?<=[A-Za-z])\s*点\s*(?=[A-Za-z])")
# 单独念域名后缀: "点com" -> ".com" (前面是开头/标点/空格/字母时才转)
RE_TLD = r"com|cn|net|org|io|ai|cc|top|me|tv|info|xyz"
TLD_SET = set(RE_TLD.split("|"))
RE_TLD_DOT = re.compile(r"(^|[，,。！？；；、\s])\s*点\s*(%s)(?![A-Za-z])" % RE_TLD,
                        re.I)
# "三w" 是中文里念 www 的常见说法
RE_THREE_W = re.compile(r"三\s*[wW](?![A-Za-z])")
# "点儿" 是 "点" 的儿化音(一点儿、有点儿、差点儿 都等价于 点)
RE_ERHUA = re.compile(r"点儿")
# 整段域名: 允许中文标签, 例如 www点百度点com
RE_DOMAIN_CHUNK = re.compile(
    r"^[A-Za-z0-9\u4e00-\u9fff-]+(?:\s*[点.．]\s*[A-Za-z0-9\u4e00-\u9fff-]+)+$")
# 字母+中文数字必须紧挨着, 避免把 "post 一下" 误改成 "POST1下"
RE_LETTER_NUM = re.compile(r"(?<![A-Za-z])([A-Za-z]{1,3})([一二三四五六七八九十])(?![一二三四五六七八九十])")
RE_PERCENT = re.compile(r"百分之\s*([%s点]+)" % ALL_CN)
RE_YEAR = re.compile(r"(?<![%s])([%s]{4})\s*年" % (ALL_CN, ALL_CN))
RE_MONTH = re.compile(r"(?<![%s])(?P<num>[%s]{1,3})\s*(?P<unit>月|日|号)" % (ALL_CN, ALL_CN))
RE_MONEY = re.compile(r"(?P<num>[%s]+)\s*(?P<unit>元|块钱|块)(?!儿)(?=$|[，。！？；：、,.!?;:\s])"
                      % ALL_CN)
RE_TIME = re.compile(r"(?P<h>[%s]{1,3})\s*点\s*(?P<rest>半|[%s]{1,3}\s*分)" % (ALL_CN, ALL_CN))
RE_PERIOD_TIME = re.compile(
    r"(?P<per>上午|下午|晚上|中午|早上|凌晨)\s*(?P<h>[%s]{1,3})\s*点\s*(?P<rest>半|[%s]{1,3}\s*分)?"
    % (ALL_CN, ALL_CN))
# 口语省略式: 系数 + 万/千/百 + 单个光数字, 如 一万二 / 四千五 / 一百五
# 口语省略式: 系数 + 万/千/百 + 单个光数字, 如 一万二 / 四千五 / 一百五 / 十万八
# (系数本身也可以是单位: 十万八里的"十")
RE_TRUNC = re.compile(r"^(.*[一二两三四五六七八九零〇十百千][万千百])([一二两三四五六七八九])$")

# 重复词(白名单, 只收口述里真正会结巴的词)
RE_REPEAT = re.compile(
    r"((?:我们|你们|他们|就是|然后|这个|那个|但是|所以|因为|其实|今天|我|你|他|她|它))(?:\1)+")

# 语气词: 只删"句首/标点后 + 语气词 + 逗号"这种确定用法
RE_FILLER = re.compile(r"(^|[，。！？；：,.!?;:])\s*(?:嗯+|呃+|哦+|哎+|唉+|啊+)\s*[，,]\s*")
# 句首/标点后的"那个，" "就是说，"
RE_DISCOURSE = re.compile(r"(^|[，。！？；：])\s*(?:那个|就是说)\s*[，,]\s*")

# 整句/整段只有一个数字: "三千两百八十九。" -> "3289。" "幺八八幺幺幺" -> "188111"
# 只有"整段都是数字"时才转, 句子里夹着汉字(如"我买了三千两百八十九个")一律不动
CN_DIGIT_EXT = dict(CN_DIGIT)
CN_DIGIT_EXT["幺"] = 1
ALL_CN_EXT = ALL_CN + "幺"
RE_STAND_NUM = re.compile(r"^([%s]+)$" % ALL_CN_EXT)
RE_STAND_DEC = re.compile(r"^([%s]+)点([%s]+)$" % (ALL_CN_EXT, "".join(CN_DIGIT_EXT)))
RE_SEGMENT = re.compile(r"([，,。！？；;、\n])")
# 逐位念的数字串: 连续 3 个以上数字字符(不含零). 正常数词里数字和单位是交替的,
# 所以 "一二三" "五六七八" 这种只可能是逐位读法, 不能整体求值(否则 "一二三" 会变成 3)
RE_DIGIT_RUN = re.compile(r"[一二两三四五六七八九幺]{3,}")


def _cn2str(s):
    """中文数字 -> 字符串(保留逐位读法的前导零). 失败返回 None"""
    if len(s) > 1 and all(c in CN_DIGIT_EXT for c in s):
        return "".join(str(CN_DIGIT_EXT[c]) for c in s)
    if RE_DIGIT_RUN.search(s):
        # 逐位念的数字串里夹了单位字: "一二三四五六七八九十" -> 12345678910
        out = []
        for c in s:
            if c in CN_DIGIT_EXT:
                out.append(str(CN_DIGIT_EXT[c]))
            elif c in CN_UNIT:
                out.append(str(CN_UNIT[c]))
            else:
                return None
        return "".join(out)
    v = cn2int(s.replace("幺", "一"))
    if v is None:
        return None
    # "万万" 这种解析成 0 的, 视为失败, 不动
    if v == 0 and not any(c in "零〇" for c in s):
        return None
    return str(v)


def _conv_segment(seg):
    if not seg.strip():
        return seg
    m = RE_STAND_DEC.match(seg.strip())
    if m:
        a = _cn2str(m.group(1))
        b = "".join(str(CN_DIGIT_EXT[c]) for c in m.group(2))
        if a is not None:
            return f"{a}.{b}"
        return seg
    m = RE_STAND_NUM.match(seg.strip())
    if m:
        v = _cn2str(m.group(1))
        if v is not None:
            return v
    return seg


def convert_standalone_numbers(text):
    parts = RE_SEGMENT.split(text)
    return "".join(p if RE_SEGMENT.fullmatch(p) else _conv_segment(p) for p in parts)


def convert_spelled_latin(text, keep):
    parts = RE_SEGMENT.split(text)
    return "".join(p if RE_SEGMENT.fullmatch(p) else _conv_latin(p, keep) for p in parts)


def _conv_domain_chunk(seg):
    """整段域名: 中文标签转拼音. www点百度点com -> www.baidu.com"""
    t = seg.strip()
    if not RE_DOMAIN_CHUNK.match(t):
        return seg
    labels = re.split(r"\s*[点.．]\s*", t)
    if len(labels) < 2 or labels[-1].lower() not in TLD_SET:
        return seg
    if all(lb.isascii() for lb in labels):
        return seg          # 纯英文域名交给大小写规则处理
    if not labels[0].isascii():
        return seg          # 第一段必须是 www/英文(如 "百度点com" 保持保守)
    if lazy_pinyin is None and any(not lb.isascii() for lb in labels):
        return seg          # 没装 pypinyin 时宁可不转, 也不输出半中半英的域名
    out = []
    for lb in labels:
        if not lb.isascii() and lazy_pinyin is not None:
            lb = "".join(lazy_pinyin(lb))
        out.append(lb.lower())
    return ".".join(out)


def convert_domains(text):
    parts = RE_SEGMENT.split(text)
    return "".join(p if RE_SEGMENT.fullmatch(p) else _conv_domain_chunk(p) for p in parts)


RE_DUP_COMMA = re.compile(r"[，,]{2,}")
RE_DUP_STOP = re.compile(r"。{2,}")
RE_COMMA_STOP = re.compile(r"[，,]\s*。")
RE_STOP_COMMA = re.compile(r"。[，,]")
RE_SPACE_PUNCT = re.compile(r"\s+([，。！？；：、])")
RE_LEADING = re.compile(r"^[，,。！？；：、\s]+")
# 网址结尾的句号去掉: "www.baidu.com。" -> "www.baidu.com"
# 只认"点 + 已知后缀"结尾, 正常句子的句号不受影响
RE_URL_TAIL = re.compile(r"(\.(?:%s))[。.．]$" % RE_TLD, re.I)


def _sub_spaced(m):
    # 逐字母念出来的序列统一转小写: "H I O U" -> "hiou"
    # (热词在之后还会把已知词的大小写改回来, 例如 stripchat -> Stripchat)
    return re.sub(r"[ \t]+", "", m.group(1)).lower()


def _sub_letter_num(m):
    return m.group(1).upper() + str(CN_DIGIT[m.group(2)])


# 大小写混乱的短英文串(逐字母拼读的典型产物): "HIou" -> "hiou"
# 这些品牌/写法保持原样, 不改小写
DEFAULT_KEEP_CASE = {
    "iPhone", "iPad", "iPod", "iOS", "macOS", "iMac", "iCloud", "iTunes",
    "eBay", "YouTube", "GitHub", "GitLab", "JavaScript", "TypeScript",
    "PowerShell", "OpenAI", "ChatGPT", "DeepSeek", "FireRedASR", "Typora",
    "PyTorch", "TensorFlow", "WeChat", "WeBox", "MySQL", "PostgreSQL",
    "WiFi", "Bluetooth", "LinkedIn", "TikTok", "WhatsApp", "PlayStation",
    "GeForce", "MacBook", "AirPods", "WordPress", "LaTeX", "eSIM",
}
RE_LATIN_TOKEN = re.compile(r"^[A-Za-z]{2,12}$")


def _mixed_case(tok):
    """True = 大小写既不是全大写/全小写, 也不是首字母大写"""
    if tok.isupper() or tok.islower():
        return False
    return not (tok[0].isupper() and tok[1:].islower())


RE_DOMAIN = re.compile(r"^[A-Za-z]{1,12}(\.[A-Za-z]{1,8})+$")
# "HIou." 这种句末带 ASCII 标点的, 把标点摘出来单独保留
RE_LATIN_TAIL = re.compile(r"^([A-Za-z][A-Za-z.]*?)([.!?,;:'\"]*)$")


def _conv_latin(seg, keep):
    t = seg.strip()
    m = RE_LATIN_TAIL.match(t)
    if not m:
        return seg
    core, tail = m.group(1), m.group(2)
    # 域名逐段处理: StrIpchat.com -> stripchat.com, 但 GitHub.com 保持
    if RE_DOMAIN.match(core):
        out, changed = [], False
        for lb in core.split("."):
            if len(lb) >= 2 and _mixed_case(lb) and lb.lower() not in keep:
                out.append(lb.lower())
                changed = True
            else:
                out.append(lb)
        return ".".join(out) + tail if changed else seg
    if not RE_LATIN_TOKEN.match(core) or not _mixed_case(core):
        return seg
    if core.lower() in keep:
        return seg
    return core.lower() + tail


def _sub_year(m):
    s = m.group(1)
    if not all(c in CN_DIGIT for c in s):
        return m.group(0)
    return "".join(str(CN_DIGIT[c]) for c in s) + "年"


# ---------------- 末尾标点 ----------------

_TRAIL_BASE = "。．.，,、；;…～~？！?!"    # 所有可能出现在句尾的标点
_TRAIL_RE_CACHE = {}


def _trail_re(keep):
    r = _TRAIL_RE_CACHE.get(keep)
    if r is None:
        chars = [re.escape(c) for c in _TRAIL_BASE if c not in keep] + ["\\s"]
        r = re.compile("[" + "".join(chars) + "]+$")
        _TRAIL_RE_CACHE[keep] = r
    return r


def strip_trailing_punct(text, cfg):
    """短句末尾的句号类标点没必要(“你好世界。” -> “你好世界”), 长句保留。
    但问号/叹号/省略号/波浪号承载含义和语气(“他走了。” vs “他走了？”),
    默认永不删, 可用 trailing_punct_keep 自定义(设为空串则全去)。
    trailing_punct: auto=短句去/长句留; strip=总去(仍受 keep 保护); keep=总留"""
    mode = str(cfg.get("trailing_punct") or "keep").strip().lower()
    if mode not in ("auto", "strip"):
        return text            # keep/未知值/缺省: 一律不动(保守)
    if mode == "auto":
        try:
            limit = float(cfg.get("trailing_punct_max_len", 15))
        except Exception:
            limit = 15.0
        if len(text.strip()) > limit:
            return text
    keep = str(cfg.get("trailing_punct_keep") if cfg.get("trailing_punct_keep") is not None
               else "？！…～~")
    stripped = _trail_re(keep).sub("", text)
    return stripped or text


# ---------------- 自定义替换 ----------------

_rules_cache = {"path": None, "mtime": None, "rules": []}


def load_rules(path, logger=None):
    """读取 replacements.txt: 每行 '原词 => 新词', # 开头注释. 按原词长度降序"""
    try:
        if not path or not os.path.exists(path):
            return []
        mt = os.path.getmtime(path)
        if _rules_cache["path"] == path and _rules_cache["mtime"] == mt:
            return _rules_cache["rules"]
        rules = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=>" not in line:
                    continue
                a, b = line.split("=>", 1)
                a, b = a.strip(), b.strip()
                if a:
                    rules.append((a, b))
        rules.sort(key=lambda x: -len(x[0]))
        for a, b in rules:
            if a.isdigit() and logger:
                logger(f"替换规则里的纯数字可能误伤: {a} => {b}")
        _rules_cache.update(path=path, mtime=mt, rules=rules)
        return rules
    except Exception:
        return []


def _apply_rules(text, rules):
    for a, b in rules:
        if a in text:
            text = text.replace(a, b)
    return text


# ---------------- 主入口 ----------------

def apply(text, cfg=None, rules=None):
    cfg = cfg or {}
    if not text or not cfg.get("postprocess", True):
        return text

    out = text

    # 1. 被空格拆散的英文: "S t r I p c h a t" -> "Stripchat"
    try:
        out = RE_ERHUA.sub("点", out)
        out = RE_THREE_W.sub("www", out)
        out = RE_SPACED.sub(_sub_spaced, out)
        out = RE_DOTCOM.sub(".com", out)
        out = RE_CN_DOT.sub(".", out)
        out = RE_TLD_DOT.sub(lambda m: m.group(1) + "." + m.group(2).lower(), out)
        out = convert_domains(out)
        keep = {x.lower() for x in DEFAULT_KEEP_CASE}
        keep |= {str(x).lower() for x in (cfg.get("latin_keep_case") or [])}
        out = convert_spelled_latin(out, keep)
    except Exception:
        pass

    # 2. 字母+中文数字: "f九" -> "F9", "gpt四" -> "GPT4"
    try:
        out = RE_LETTER_NUM.sub(_sub_letter_num, out)
    except Exception:
        pass

    # 3. 固定缩写大小写: po -> PO (只在独立出现时)
    try:
        for abbr in cfg.get("uppercase_abbr") or []:
            if abbr:
                out = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(abbr),
                             abbr.upper(), out, flags=re.I)
    except Exception:
        pass

    # 4. 数字/日期/金额/百分比/时间
    if cfg.get("number_norm", True):
        try:
            out = RE_PERCENT.sub(_sub_percent, out)
            out = RE_YEAR.sub(_sub_year, out)
            out = RE_MONTH.sub(_sub_num, out)
            out = RE_MONEY.sub(_sub_num, out)
            out = RE_PERIOD_TIME.sub(_sub_period_time, out)
            out = RE_TIME.sub(_sub_time, out)
            out = convert_standalone_numbers(out)
        except Exception:
            pass

    # 5. 自定义替换(用户确认过的稳定错法)
    try:
        if rules:
            out = _apply_rules(out, rules)
    except Exception:
        pass

    # 6. 口语顺滑
    if cfg.get("filler_removal", True):
        try:
            out = RE_REPEAT.sub(r"\1", out)
            out = RE_DISCOURSE.sub(r"\1", out)
            out = RE_FILLER.sub(r"\1", out)
        except Exception:
            pass

    # 7. 标点清理
    try:
        out = RE_DUP_COMMA.sub("，", out)
        out = RE_DUP_STOP.sub("。", out)
        out = RE_COMMA_STOP.sub("。", out)
        out = RE_STOP_COMMA.sub("。", out)
        out = RE_SPACE_PUNCT.sub(r"\1", out)
        out = RE_LEADING.sub("", out)
        out = RE_URL_TAIL.sub(r"\1", out)
        out = out.strip()
    except Exception:
        pass

    # 8. 末尾标点: 对话里的简短提问不带句号更自然
    try:
        out = strip_trailing_punct(out, cfg)
    except Exception:
        pass

    return out or text
