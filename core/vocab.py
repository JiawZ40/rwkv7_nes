"""Vocabulary & control-tag definition for the RWKV-7 causal MIDI loop generator.

Token stream layout (REMI-like, explicit Note-On / Note-Off so that a loop can be
closed deterministically):

    [BOS] <MOOD> <TONE> <SPEED> <DENSITY> <INST>
    [BAR] [POS_i] [NOTE_OFF_p ...] [NOTE_ON_p ...]   x 4 bars
    [EOS]
"""

from __future__ import annotations

PITCHES = 128
POS_PER_BAR = 16          # 4/4 at 16th-note resolution
BARS = 4                  # [BAR_LENGTH_4]

# --------------------------------------------------------------------------- #
# control tag spaces
# --------------------------------------------------------------------------- #
MOODS = ["MOOD_EPIC_TENSION", "MOOD_HEROIC", "MOOD_CALM", "MOOD_MYSTERY"]
TONES = [f"TONE_{r}_{m}" for m in ("MINOR", "MAJOR") for r in
         ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]]
SPEEDS = ["SPEED_SLOW", "SPEED_MEDIUM_100", "SPEED_FAST_130"]
DENSITIES = ["DENSITY_LOW", "DENSITY_MEDIUM", "DENSITY_HIGH"]
INSTS = ["INST_BRASS_SECTION", "INST_PULSE_LEAD", "INST_TRIANGLE_BASS", "INST_NOISE_PERC"]

NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
NOTE_PC = {n: i for i, n in enumerate(NOTE_NAMES)}
NOTE_LABEL = {"C": "C", "Cs": "C♯", "D": "D", "Ds": "D♯", "E": "E", "F": "F",
              "Fs": "F♯", "G": "G", "Gs": "G♯", "A": "A", "As": "A♯", "B": "B"}

# 每个 INST 标签的音域与复音上限（NES 声源特征：三角波低音、脉冲主旋律、
# 铜管/噪声声部音区更高），控制解码时的音高先验与最大同时发声数。
# NOISE_PERC 是 GM 打击乐映射：35-81 之外的音高不是鼓，参见 Theory.is_perc。
INST_RANGES = {
    "INST_BRASS_SECTION": (43, 84),
    "INST_PULSE_LEAD": (55, 88),
    "INST_TRIANGLE_BASS": (26, 55),
    "INST_NOISE_PERC": (35, 81),
}
INST_POLY = {
    "INST_BRASS_SECTION": 4,
    "INST_PULSE_LEAD": 2,
    "INST_TRIANGLE_BASS": 2,
    "INST_NOISE_PERC": 4,          # 鼓可以密集叠置（底鼓+军鼓+踩镲）
}

# GM 打击乐映射里组成一个可用套鼓的那些件，解码时给偏好。
# 未列出的 GM 音高（沙锤、刮瓜、长笛类音效等）不代表「错的」，但一套
# 能听的节奏骨架必须建立在底鼓/军鼓/踩镲上，所以给很低的权重压下去。
GM_DRUM_CORE = {
    35: 1.5, 36: 1.8,                     # 底鼓
    38: 1.7, 40: 1.3, 37: 0.9,            # 军鼓 / 边击
    42: 1.8, 44: 1.5, 46: 1.2,            # 闭镲 / 脚踏镲 / 开镲
    41: 0.9, 43: 0.9, 45: 0.9, 47: 0.9, 48: 0.9, 50: 0.9,   # 桶鼓
    49: 0.7, 51: 0.7, 57: 0.7, 59: 0.7,   # 吊镲 / 叮叮镲
}
GM_DRUM_OTHER = 0.25                      # 其余 GM 打击乐音效


def parse_tone(tag: str):
    """TONE_<Root>_<MODE> -> (root_pc, mode)   (默认 D minor)"""
    root_pc, mode = 2, "MINOR"
    try:
        parts = tag.split("_")
        if len(parts) >= 3 and parts[0] == "TONE":
            root_pc = NOTE_PC.get(parts[1], 2)
            mode = parts[2].upper()
            if mode not in ("MINOR", "MAJOR"):
                mode = "MINOR"
    except Exception:
        pass
    return root_pc, mode


def scale_pcs(root_pc: int, mode: str = "MINOR"):
    """调内音级集合（pitch class）"""
    intervals = (0, 2, 3, 5, 7, 8, 10) if mode == "MINOR" else (0, 2, 4, 5, 7, 9, 11)
    return {(root_pc + i) % 12 for i in intervals}


class Vocab:
    def __init__(self):
        self.itos, self.stoi = [], {}

        def add(tok):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

        for t in ["<PAD>", "<BOS>", "<EOS>"]:
            add(t)
        for group in (MOODS, TONES, SPEEDS, DENSITIES, INSTS):
            for t in group:
                add(t)
        add("BAR")
        for i in range(POS_PER_BAR):
            add(f"POS_{i}")
        for p in range(PITCHES):
            add(f"ON_{p}")
        for p in range(PITCHES):
            add(f"OFF_{p}")

        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.bar_id = self.stoi["BAR"]
        self.pos_ids = [self.stoi[f"POS_{i}"] for i in range(POS_PER_BAR)]
        self.on_ids = [self.stoi[f"ON_{p}"] for p in range(PITCHES)]
        self.off_ids = [self.stoi[f"OFF_{p}"] for p in range(PITCHES)]
        self.ctrl_ids = [self.stoi[t] for g in (MOODS, TONES, SPEEDS, DENSITIES, INSTS)
                         for t in g]

    # ---------------------------------------------------------------- #
    def __len__(self):
        return len(self.itos)

    def enc(self, tok):
        return self.stoi[tok]

    def dec(self, i):
        return self.itos[i]

    def encode(self, toks):
        return [self.stoi[t] for t in toks]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


# --------------------------------------------------------------------------- #
# music-theory helpers
# --------------------------------------------------------------------------- #
# D natural minor (requested tonality [TONE_D_MINOR]): D E F G A Bb C
D_MINOR_PCS = {2, 4, 5, 7, 9, 10, 0}
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_of(pitch_hist):
    """Krumhansl-Schmuckler key detection. pitch_hist: len-12 pitch-class weights."""
    import numpy as np
    h = np.asarray(pitch_hist, dtype=float)
    if h.sum() <= 0:
        return "C", "MAJOR", -1.0
    h = h / h.sum()
    best = (None, None, -1e9)
    for mode, prof in (("MAJOR", MAJOR_PROFILE), ("MINOR", MINOR_PROFILE)):
        prof = np.asarray(prof, dtype=float)
        for root in range(12):
            r = np.roll(prof, root)
            c = float(np.corrcoef(h, r)[0, 1])
            if c > best[2]:
                best = (NOTE_NAMES[root], mode, c)
    return best


def is_in_scale(pitch, root_pc=2, mode="MINOR"):
    """scale membership (natural minor / major)"""
    interval = (pitch - root_pc) % 12
    if mode == "MINOR":
        return interval in {0, 2, 3, 5, 7, 8, 10}
    return interval in {0, 2, 4, 5, 7, 9, 11}


def in_scale_pitches(root_pc=2, mode="MINOR", lo=36, hi=96):
    return [p for p in range(lo, hi + 1) if is_in_scale(p, root_pc, mode)]
