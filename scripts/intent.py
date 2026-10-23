# -*- coding: utf-8 -*-
"""intent.py —— 把「用户想听什么」变成小模型看得懂的控制标签

两条路：

1) 宿主 LLM（推荐）：读 SKILL.md 里的 SPEC_SCHEMA，直接产出 spec JSON，
   交给 bgm.py --spec。LLM 的语义理解远好于关键词，能处理
   「那种在雪山顶上走路、有点孤独但又要往前走的感觉」这类描述。

2) 规则兜底（本文件）：没有 LLM / LLM 不想出 JSON 时用 `--text`，
   按场景词 + 情绪词映射到预设。覆盖常见游戏场景，够用但不是重点。

spec 字段（全部可选，缺失的按 preset 或默认值补）：

    name     输出文件名（英文/拼音，不要空格）
    style    "nes"（芯片编排，默认）| "pop"（抒情编排）
    sound    "2a03"（自研 FC 合成器，默认）| "sf2"（FluidSynth + 音色库）
    mix      "full"（加倍+合唱+饱和+总线压缩混响，默认）| "wide"（轻加宽）
             | "chip"（原味干声，FC 直出）
    root     调根音："C" "Cs" "D" ... 或 0-11
    mode     "minor" | "major"
    mood     "heroic" | "epic_tension" | "calm" | "mystery"
    speed    "slow" | "medium" | "fast"
    density  "low" | "medium" | "high"
    bpm      速度（不给就用场景预设）
    bars     总小节数（4 的倍数，默认 32）
    seed     随机种子（同一个 seed 结果可复现）
    n_cand   每个声部抽几张卡（默认 3，越大越慢也越挑）
    motif    动机 [[start,end,pitch], ...]，16 分音符步 0-63，模型据此续写
"""
from __future__ import annotations

import json
import os
import re

NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
ALIAS = {"Db": "Cs", "Eb": "Ds", "Gb": "Fs", "Ab": "Gs", "Bb": "As",
         "C#": "Cs", "D#": "Ds", "F#": "Fs", "G#": "Gs", "A#": "As"}

# --------------------------------------------------------------------------- #
# 游戏场景预设：一句话 -> 一整套可用的编曲参数
# --------------------------------------------------------------------------- #
# key: 场景名  value: (中文/英文关键词, 参数)
PRESETS = {
    "overworld": (["大地图", "世界地图", "草原", "野外", "冒险", "森林", "地图",
                   "overworld", "field", "grass", "forest", "adventure"],
                  dict(mood="heroic", speed="fast", density="medium", bpm=168,
                       mode="major", style="nes")),
    "underworld": (["地下", "洞窟", "地牢", "矿洞", "熔岩", "洞穴", "地下城",
                    "underworld", "cave", "dungeon", "mine", "lava"],
                   dict(mood="mystery", speed="medium", density="medium",
                        bpm=132, mode="minor", style="nes")),
    "castle": (["城堡", "要塞", "神殿", "遗迹", "迷宫", "古堡",
                "castle", "fortress", "temple", "ruin", "maze"],
               dict(mood="epic_tension", speed="fast", density="high", bpm=150,
                    mode="minor", style="nes")),
    "boss": (["boss", "头目", "决战", "最终关", "魔王", "战斗", "打斗",
              "boss", "final", "battle"],
             dict(mood="epic_tension", speed="fast", density="high", bpm=176,
                  mode="minor", style="nes")),
    "underwater": (["水下", "海底", "水底", "游泳",
                    "underwater", "water", "ocean"],
                   dict(mood="calm", speed="medium", density="low", bpm=112,
                        mode="major", style="nes")),
    "town": (["城镇", "村庄", "商店", "酒馆", "村庄", "菜单", "村子",
              "town", "village", "shop", "menu", "inn"],
             dict(mood="calm", speed="medium", density="medium", bpm=120,
                  mode="major", style="nes")),
    "victory": (["胜利", "通关", "结算", "得分", "奖励", "升段",
                 "victory", "win", "clear", "fanfare"],
                dict(mood="heroic", speed="fast", density="high", bpm=160,
                     mode="major", style="nes")),
    "gameover": (["失败", "死亡", "结束", "gameover", "game over", "die",
                  "失败画面", "dead"],
                 dict(mood="epic_tension", speed="slow", density="low", bpm=84,
                      mode="minor", style="nes")),
    "title": (["标题", "开场", "主菜单", "片头", "title", "opening", "intro",
               "logo"],
              dict(mood="heroic", speed="medium", density="medium", bpm=140,
                   mode="minor", style="nes")),
    "shoot": (["射击", "机甲", "太空", "赛博", "飞船", "弹幕",
               "shoot", "space", "cyber", "mecha", "space"],
              dict(mood="epic_tension", speed="fast", density="high", bpm=184,
                   mode="minor", style="nes")),
    "puzzle": (["解谜", "思考", "推箱子", "机关", "puzzle", "think"],
               dict(mood="mystery", speed="slow", density="low", bpm=100,
                    mode="minor", style="nes")),
    "credits": (["片尾", "制作人员", "结尾", "credits", "ending", "staff"],
                dict(mood="calm", speed="slow", density="low", bpm=92,
                     mode="major", style="pop")),
    "ballad": (["抒情", "流行", "情歌", "伤感", "安静", "钢琴",
                "ballad", "pop", "sad", "emotional"],
               dict(mood="calm", speed="slow", density="low", bpm=92,
                    mode="minor", style="pop")),
}
DEFAULT_PRESET = "overworld"

# 情绪/速度/密度的兜底词（场景没命中时用）
MOOD_WORDS = {
    "heroic": ["英雄", "热血", "燃", "激昂", "勇者", "冲锋", "heroic", "hero",
               "epic", "brave", "triumph"],
    "epic_tension": ["紧张", "压迫", "危机", "战斗", "暗黑", "史诗", "悬疑",
                     "tension", "tense", "dark", "danger", "suspense"],
    "calm": ["平静", "轻松", "温柔", "悠闲", "治愈", "安静", "舒服",
             "calm", "peaceful", "relax", "gentle", "soft"],
    "mystery": ["神秘", "诡异", "未知", "迷幻", "孤独", "空灵",
                "mystery", "mysterious", "eerie", "lonely", "strange"],
}
SPEED_WORDS = {"fast": ["快", "急促", "高速", "紧张", "fast", "quick", "rapid"],
               "slow": ["慢", "缓慢", "悠", "舒缓", "slow", "soft"],
               "medium": ["中速", "中等", "medium"]}
DENSITY_WORDS = {"high": ["密集", "复杂", "华丽", "满", "high", "dense", "busy"],
                 "low": ["稀疏", "简单", "留白", "空", "low", "sparse", "simple"],
                 "medium": ["适中", "medium"]}
MODE_WORDS = {"major": ["大调", "明亮", "开心", "阳光", "major", "bright",
                        "happy"],
              "minor": ["小调", "暗", "悲伤", "忧郁", "minor", "dark", "sad"]}
# 带调式的写法优先（"D小调" / "C major" / "Am"），其次才是光秃秃的字母，
# 且字母前后都不能是英文字母，否则 "Boss" 会被当成 B 调
KEYED_RE = re.compile(r"([A-G])([#b]?)\s*(大调|小调|major|minor|maj|min|m)(?![A-Za-z])",
                      re.I)
BARE_RE = re.compile(r"(?<![A-Za-z])([A-G])([#b]?)(?![A-Za-z])")


def norm_root(v, default=9):
    """任意写法 -> pitch class 0-11"""
    if v is None:
        return default
    if isinstance(v, int):
        return v % 12
    s = str(v).strip()
    if s.isdigit():
        return int(s) % 12
    s = ALIAS.get(s, s)
    base = s[:1].upper()
    if base not in NOTE_NAMES:
        return default
    return NOTE_NAMES.index(base)


def norm_spec(spec):
    """补全 / 夹紧 spec，产出 bgm.py 直接能吃的形式"""
    s = dict(spec or {})
    preset = PRESETS.get(str(s.get("scene", "")).lower(), (None, None))[1]
    if preset:
        for k, v in preset.items():
            s.setdefault(k, v)
    else:
        d = PRESETS[DEFAULT_PRESET][1]
        for k, v in d.items():
            s.setdefault(k, v)

    s["style"] = str(s.get("style", "nes")).lower()
    if s["style"] not in ("nes", "pop"):
        s["style"] = "nes"
    s["sound"] = str(s.get("sound", "2a03")).lower()
    if s["sound"] not in ("2a03", "sf2"):
        s["sound"] = "2a03"
    s["mix"] = str(s.get("mix", "full")).lower()
    if s["mix"] not in ("chip", "wide", "full"):
        s["mix"] = "full"

    s["mood"] = str(s.get("mood", "heroic")).lower()
    if s["mood"] not in ("heroic", "epic_tension", "calm", "mystery"):
        s["mood"] = "heroic"
    s["speed"] = str(s.get("speed", "fast")).lower()
    if s["speed"] not in ("slow", "medium", "fast"):
        s["speed"] = "fast"
    s["density"] = str(s.get("density", "medium")).lower()
    if s["density"] not in ("low", "medium", "high"):
        s["density"] = "medium"
    s["mode"] = str(s.get("mode", "minor")).lower()
    if s["mode"] not in ("minor", "major"):
        s["mode"] = "minor"

    s["root_pc"] = norm_root(s.get("root"), 9)
    s["bpm"] = max(60, min(220, int(s.get("bpm", 150))))
    s["bars"] = max(4, min(128, int(s.get("bars", 32)) // 4 * 4))
    s["seed"] = int(s.get("seed", 7))
    s["n_cand"] = max(1, min(8, int(s.get("n_cand", 3))))
    s["name"] = re.sub(r"[^A-Za-z0-9_\-]", "_", str(s.get("name", "bgm")))[:48]
    motif = s.get("motif")
    s["motif"] = motif if isinstance(motif, list) and motif else None
    return s


def from_text(text):
    """规则兜底：自然语言 -> spec（命中不到就走默认场景）"""
    t = (text or "").lower()
    scene, best_hit = None, 0.0
    for key, (words, _cfg) in PRESETS.items():
        # 长词权重更高：「水下关卡」里「水下」比「关卡」更具体
        hits = sum(len(w) for w in words if w.lower() in t)
        if hits > best_hit:
            scene, best_hit = key, hits

    def first(mapping, default):
        for k, ws in mapping.items():
            if any(w in t for w in ws):
                return k
        return default

    cfg = dict(PRESETS.get(scene, PRESETS[DEFAULT_PRESET])[1])
    cfg["scene"] = scene or DEFAULT_PRESET
    cfg["mood"] = first(MOOD_WORDS, cfg["mood"])
    cfg["speed"] = first(SPEED_WORDS, cfg["speed"])
    cfg["density"] = first(DENSITY_WORDS, cfg["density"])
    cfg["mode"] = first(MODE_WORDS, cfg["mode"])

    m = KEYED_RE.search(text or "") or BARE_RE.search(text or "")
    if m:
        r = ALIAS.get(m.group(1).upper() + m.group(2), m.group(1).upper())
        if r in NOTE_NAMES:
            cfg["root"] = NOTE_NAMES.index(r)
        suffix = (m.group(3) or "").lower()
        if suffix in ("maj", "major", "大调"):
            cfg["mode"] = "major"
        elif suffix in ("m", "min", "minor", "小调"):
            cfg["mode"] = "minor"
    mbpm = re.search(r"(\d{2,3})\s*(bpm|拍)", t)
    if mbpm:
        cfg["bpm"] = int(mbpm.group(1))
    mbar = re.search(r"(\d+)\s*(小节|bars?|bar)", t)
    if mbar:
        cfg["bars"] = int(mbar.group(1)) // 4 * 4
    return norm_spec(cfg)


SPEC_EXAMPLE = {
    "name": "castle_boss",
    "scene": "boss",
    "style": "nes",
    "sound": "2a03",
    "root": "D",
    "mode": "minor",
    "mood": "epic_tension",
    "speed": "fast",
    "density": "high",
    "bpm": 168,
    "bars": 32,
    "seed": 7,
}


def dump_schema(path=None):
    doc = {
        "note": "宿主 LLM 按此 schema 产出 spec，再调用 scripts/bgm.py --spec",
        "example": SPEC_EXAMPLE,
        "scenes": sorted(PRESETS.keys()),
        "fields": {
            "style": ["nes", "pop"],
            "sound": ["2a03", "sf2"],
            "mix": ["full", "wide", "chip"],
            "mood": ["heroic", "epic_tension", "calm", "mystery"],
            "speed": ["slow", "medium", "fast"],
            "density": ["low", "medium", "high"],
            "mode": ["minor", "major"],
            "root": "C Cs D Ds E F Fs G Gs A As B（或 0-11）",
            "bars": "4 的倍数，默认 32（约 30-60 秒）",
        },
    }
    txt = json.dumps(doc, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
    return txt


def load_spec(src):
    """--spec 可以传文件路径，也可以直接传 JSON 字符串"""
    if not src:
        return {}
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(src)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--schema":
        print(dump_schema())
    else:
        print(json.dumps(from_text(" ".join(sys.argv[1:])), ensure_ascii=False,
                         indent=2))
