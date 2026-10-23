# -*- coding: utf-8 -*-
"""compose.py —— spec -> 多轨 MIDI

流程（全部串行 CPU 推理，不需要显卡）：

    控制标签   ──►  小模型抽卡（每个声部 N 张候选，4 小节一条）
                        │
                        ▼
                    乐理评分（和弦契合 / 级进 / 节奏稳定 / 音域）
                        │
                        ▼
                    段落表编排（intro / A / B / A' / outro）
                        │
                        ▼
                    4 声部 → MIDI（可编辑的工程文件）+ 渲染用的 layers

模型只在「写动机」这一步起作用，和声、织体、配器是确定性规则，
所以输出永远协和，且同一个 seed 可以复现。
"""
from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

STEPS = 64                       # 一条素材 = 4 小节 x 16 分音符
PITCH_A4 = 69.0

# --------------------------------------------------------------------------- #
# 音级 / 和声
# --------------------------------------------------------------------------- #
DEG_SEMI = {"minor": (0, 2, 3, 5, 7, 8, 10),
            "major": (0, 2, 4, 5, 7, 9, 11)}
NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

# 度数用「音阶级号」，换调式自动正确
PROGS = {
    "minor": {
        "A": [(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")],   # i-VI-III-VII
        "B": [(5, "maj"), (6, "maj"), (0, "min"), (0, "min")],   # VI-VII-i
        "C": [(3, "min"), (4, "maj"), (0, "min"), (0, "min")],   # iv-V-i
    },
    "major": {
        "A": [(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],   # I-V-vi-IV
        "B": [(3, "maj"), (4, "maj"), (5, "min"), (0, "maj")],   # IV-V-vi-I
        "C": [(1, "min"), (4, "maj"), (0, "maj"), (0, "maj")],   # ii-V-I
    },
}
MOOD_PROG = {"heroic": ("A", "B"), "epic_tension": ("A", "B"),
             "calm": ("A", "C"), "mystery": ("C", "A")}


def chord_of(root_pc, mode, prog, bar):
    """-> (名字, 根音 pitch-class, 和弦音 pitch-class 集合)"""
    deg, quality = prog[bar % len(prog)]
    semi = DEG_SEMI[mode][deg]
    root = (root_pc + semi) % 12
    ivs = (0, 3, 7) if quality == "min" else (0, 4, 7)
    pcs = {(root + i) % 12 for i in ivs}
    name = NOTE_NAMES[root] + ("m" if quality == "min" else "")
    return name, root, pcs


def chord_tones(pcs, lo=60):
    """把和弦音级放进 lo..lo+11，升序返回"""
    return sorted(lo + ((p - lo) % 12) for p in pcs)


# --------------------------------------------------------------------------- #
# 后处理（从 web/server.py 搬来，去掉 web 依赖）
# --------------------------------------------------------------------------- #
def thin_notes(notes, density_target):
    """按每小节起音密度抽稀，同时尽量保住音域铺展"""
    if not notes:
        return notes
    per_bar = max(1, int(round(density_target)))
    bars = {}
    for n in notes:
        bars.setdefault(int(n[0]) // 16, []).append(n)
    out = []
    for _bar, items in bars.items():
        if len(items) <= per_bar:
            out.extend(items)
            continue
        strong = sorted([n for n in items if n[0] % 4 == 0], key=lambda n: n[0])
        weak = sorted([n for n in items if n[0] % 4 != 0], key=lambda n: n[0])
        picked = strong[: max(1, per_bar // 2)]
        pool = list(weak) + list(strong[max(1, per_bar // 2):])
        while len(picked) < per_bar and pool:
            if not picked:
                best = pool.pop(0)
            else:
                ps = [p for _, _, p in picked]
                lo_, hi_ = min(ps), max(ps)

                def far(n):
                    return (-max(lo_ - n[2], n[2] - hi_, 0), n[0])
                pool.sort(key=far)
                best = pool.pop(0)
            picked.append(best)
        out.extend(picked)
    out.sort()
    return out


def fit_register(notes, lo, hi, pcs=None):
    """把声部整体八度平移到 [lo,hi]（八度整数倍 -> 音级不变）"""
    if not notes:
        return notes
    ps = [p for _, _, p in notes]
    cur_c = (min(ps) + max(ps)) / 2.0
    tgt_c = (lo + hi) / 2.0
    base = int(round((tgt_c - cur_c) / 12.0)) * 12
    best_shift, best_hits = base, -1
    for cand in (base - 12, base, base + 12):
        hits = sum(1 for p in ps if lo <= p + cand <= hi)
        if hits > best_hits:
            best_shift, best_hits = cand, hits
    shift = best_shift
    pcs = set(pcs) if pcs else {0, 2, 3, 5, 7, 8, 10, 11}
    in_key = [x for x in range(lo, hi + 1) if x % 12 in pcs]
    out = []
    for s, e, p in notes:
        q = p + shift
        if lo <= q <= hi:
            out.append((s, e, q))
            continue
        cand = [x for x in range(lo, hi + 1) if x % 12 == q % 12]
        if cand:
            out.append((s, e, min(cand, key=lambda x: (abs(x - q), x))))
        elif in_key:
            out.append((s, e, min(in_key, key=lambda x: (abs(x - q), x))))
        else:
            out.append((s, e, max(lo, min(hi, q))))
    out.sort()
    return out


def snap_to_key(notes, pcs):
    """调外音就近吸附到调内音，消除跨声部的半音冲突"""
    pcs = set(pcs)
    cands = [x for x in range(0, 128) if x % 12 in pcs]
    out = []
    for s, e, p in notes:
        if p % 12 in pcs:
            out.append((s, e, p))
        else:
            out.append((s, e, min(cands, key=lambda x: (abs(x - p), x))))
    out.sort()
    return out


def clamp_gm_drums(notes):
    out = []
    for s, e, p in notes:
        q = 35 + ((int(p) - 35) % 47) if not (35 <= int(p) <= 81) else int(p)
        out.append((s, e, max(35, min(81, q))))
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# 评分
# --------------------------------------------------------------------------- #
def chord_fit(notes, prog, root_pc, mode):
    fit = tot = 0.0
    for s, e, p in notes:
        _, _, pcs = chord_of(root_pc, mode, prog, int(s) // 16)
        w = (e - s) * (2.0 if int(s) % 4 == 0 else 1.0)
        tot += w
        if int(p) % 12 in pcs:
            fit += w
    return fit / max(tot, 1e-9)


def smoothness(notes):
    ps = [p for _, _, p in notes]
    if len(ps) < 3:
        return 0.0
    iv = [abs(b - a) for a, b in zip(ps, ps[1:])]
    steps = sum(1 for x in iv if x <= 2) / len(iv)
    leaps = sum(1 for x in iv if x > 7) / len(iv)
    return steps - 1.5 * leaps


def mean_dur(notes):
    return sum(e - s for s, e, _ in notes) / max(len(notes), 1)


def score_lead(notes, prog, root_pc, mode):
    s = chord_fit(notes, prog, root_pc, mode) * 3.0
    s += smoothness(notes) * 1.2
    s += max(0.0, mean_dur(notes) - 2.0) * 0.25
    rng = max(p for _, _, p in notes) - min(p for _, _, p in notes)
    if rng > 20:
        s -= (rng - 20) * 0.1
    if not (8 <= len(notes) <= 34):
        s -= 1.5
    s -= (sum(1 for x in notes if x[1] - x[0] < 2) / len(notes)) * 1.0
    s += (sum(1 for x in notes if int(x[0]) % 2 == 0) / len(notes)) * 0.6
    return s


def score_bass(notes, prog, root_pc, mode):
    s, hits, tot = 0.0, 0, 0
    for st, _e, p in notes:
        if int(st) % 16 == 0:
            _, root, _ = chord_of(root_pc, mode, prog, int(st) // 16)
            tot += 1
            if int(p) % 12 == root:
                hits += 1
    if tot:
        s += (hits / tot) * 2.5
    s += smoothness(notes) * 0.8
    s += max(0.0, mean_dur(notes) - 2.0) * 0.15
    if not (8 <= len(notes) <= 24):
        s -= 1.0
    s += (sum(1 for x in notes if int(x[0]) % 2 == 0) / len(notes)) * 0.8
    return s


def score_drums(notes):
    s = 0.0
    for st, _e, p in notes:
        if p == 36 and int(st) % 8 == 0:
            s += 0.5
        if p == 36 and int(st) % 4 == 0:
            s += 0.2
        if p == 38 and int(st) % 8 == 4:
            s += 0.6
    kicks = sum(1 for x in notes if x[2] == 36)
    snares = sum(1 for x in notes if x[2] == 38)
    hats = sum(1 for x in notes if x[2] in (42, 44, 46))
    if 2 <= snares <= 10:
        s += 0.5
    if 4 <= kicks <= 14:
        s += 0.5
    if hats >= 6:
        s += 0.3
    if len({x[2] for x in notes}) > 7:
        s -= 1.0
    if not (12 <= len(notes) <= 30):
        s -= 1.5
    return s


# --------------------------------------------------------------------------- #
# 抽卡
# --------------------------------------------------------------------------- #
ROLE_LEAD = {"inst_tag": "INST_PULSE_LEAD", "lo": 64, "hi": 86}
ROLE_BASS = {"inst_tag": "INST_TRIANGLE_BASS", "lo": 33, "hi": 52}
ROLE_DRUM = {"inst_tag": "INST_NOISE_PERC", "lo": 35, "hi": 81, "is_perc": True}

# 芯片音乐只用得上的鼓件：底鼓 / 军鼓 / 闭镲 / 开镲 / 吊镲
NES_DRUM_KEEP = {35, 36, 38, 40, 42, 44, 46, 49}
_DRUM_CORE = sorted(NES_DRUM_KEEP)


def snap_drum(p):
    """把不在芯片鼓组里的 GM 打击乐音高归到最近的可用件"""
    return min(_DRUM_CORE, key=lambda c: (abs(c - int(p)), c))


def prep_drums(notes):
    """过滤成一套能听的芯片鼓；若整段都被滤空，就近映射而不是整段丢掉"""
    if not notes:
        return []
    keep = [n for n in notes if int(n[2]) in NES_DRUM_KEEP]
    if keep:
        return keep
    return [(s, e, snap_drum(p)) for s, e, p in notes]


def gacha(gen, role, n_cand, seed, base_tags, density, prompt=None, quiet=False):
    """对一个声部抽 n_cand 张卡，返回可用候选 [(start,end,pitch), ...]"""
    tags = tuple([t for t in base_tags if not t.startswith("INST_")]
                 + [role["inst_tag"]])
    cands, t0 = [], time.time()
    for i in range(n_cand):
        _ids, notes = gen.generate(tags, temperature=1.0,
                                   density_target=int(density),
                                   seed=int(seed) + i * 977,
                                   prompt_notes=prompt)
        if not notes:
            continue
        if role.get("is_perc"):
            notes = thin_notes(notes, density)
            notes = clamp_gm_drums(notes)
        else:
            notes = thin_notes(notes, density)
            notes = fit_register(notes, role["lo"], role["hi"])
            notes = snap_to_key(notes, base_pcs(base_tags))
        if len(notes) >= 4:
            cands.append(notes)
    if not quiet:
        print("  [gacha] %-22s %d/%d 张可用 (%.1fs)" %
              (role["inst_tag"], len(cands), n_cand, time.time() - t0))
    return cands


def base_pcs(tags):
    from vocab import parse_tone, scale_pcs
    tone = next((t for t in tags if t.startswith("TONE_")), "TONE_A_MINOR")
    root_pc, mode = parse_tone(tone)
    pcs = scale_pcs(root_pc, mode)
    if mode == "MINOR":
        pcs.add((root_pc + 9) % 12)
    return pcs


def pick(cands, scorer, label, quiet=False):
    if not cands:
        return None
    scored = sorted(((scorer(c), i, c) for i, c in enumerate(cands)), reverse=True)
    if not quiet:
        print("  [pick ] %-8s best=%.3f notes=%d" % (label, scored[0][0],
                                                     len(scored[0][2])))
    return scored[0][2]


# --------------------------------------------------------------------------- #
# 段落表
# --------------------------------------------------------------------------- #
# (名称, 和声, 主旋律卡, 有琶音, 有低音, 有鼓, 力度系数, 主旋律移八度)
TEMPLATE = [
    ("intro",  "A", None, 1, 1, 0, 0.85, 0),
    ("A1",     "A", "A",  1, 1, 1, 1.00, 0),
    ("A2",     "A", "A",  1, 1, 1, 1.00, 0),
    ("B1",     "B", "B",  1, 1, 1, 1.00, 0),
    ("B2",     "B", "B",  1, 1, 1, 1.05, 0),
    ("A3",     "A", "A",  1, 1, 1, 1.05, 12),   # 升八度 = 芯片常见的收尾
    ("outro",  "A", "A",  1, 1, 0, 0.90, 0),
]


def make_sections(n_seg):
    """按段数裁剪标准模板（保证 intro + 主体 + outro 的形状）"""
    if n_seg <= 1:
        return [("main", "A", "A", 1, 1, 1, 1.0, 0)]
    body = [TEMPLATE[1], TEMPLATE[2], TEMPLATE[3], TEMPLATE[4], TEMPLATE[5]]
    if n_seg >= len(TEMPLATE):
        secs = [TEMPLATE[0]]
        for i in range(n_seg - 2):
            secs.append(body[i % len(body)])
        secs.append(TEMPLATE[-1])
    else:
        secs = [TEMPLATE[0]]
        for i in range(n_seg - 2):
            secs.append(body[i % len(body)])
        secs.append(TEMPLATE[-1])
    return secs[:n_seg]


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
P_SQUARE = 80        # GM Lead 1 (square) —— 听感最接近 FC 脉冲通道
P_TRI = 33           # Fingered Bass（三角波低音的 GM 近似）
P_PIANO = 0
P_PAD = 91
P_STR = 48
P_GTR = 24           # 尼龙吉他
P_BASS_POP = 33


def progs_of(spec):
    """spec（调式 + 情绪 + 指定进行）-> 两套 4 小节和声进行"""
    mode = spec["mode"]
    pair = MOOD_PROG.get(spec.get("mood", "heroic"), ("A", "B"))
    if spec.get("prog") in ("A", "B", "C"):
        pair = (spec["prog"], "B" if spec["prog"] == "A" else "A")
    return {"A": PROGS[mode][pair[0]], "B": PROGS[mode][pair[1]]}


def compose(spec, gen=None, quiet=False):
    """返回 (total_steps, layers, picks, chiptune_channels, seconds_per_step)

    layers = [(stem, program, name, is_drum, notes)]，notes=(vel127, t0, t1, pitch)
    chiptune_channels 只在 style=nes 时有值，供 2A03 合成器直接使用。
    """
    style = spec.get("style", "nes")
    bpm = int(spec.get("bpm", 150))
    root_pc = spec["root_pc"]
    mode = spec["mode"]
    density = spec["density"]
    seed = int(spec.get("seed", 7))
    n_cand = max(1, int(spec.get("n_cand", 3)))
    bars = max(4, int(spec.get("bars", 32)))
    n_seg = max(1, bars // 4)

    progs = progs_of(spec)

    tags = ("MOOD_" + spec["mood"].upper(),
            "TONE_%s_%s" % (NOTE_NAMES[root_pc], mode.upper()),
            {"slow": "SPEED_SLOW", "medium": "SPEED_MEDIUM_100",
             "fast": "SPEED_FAST_130"}[spec["speed"]],
            "DENSITY_" + spec["density"].upper())

    if gen is None:
        from infer import load_generator
        gen, _ = load_generator()

    # ---- 抽卡：主旋律 A/B、低音 A/B、鼓 ---- #
    if not quiet:
        print("[compose] tags=%s bpm=%d bars=%d" % (" ".join(tags), bpm, bars))
    d_lead = {"high": 12, "medium": 9, "low": 7}[density]
    d_bass = {"high": 8, "medium": 6, "low": 5}[density]
    d_drum = {"high": 16, "medium": 13, "low": 9}[density]

    picks = {}
    for key, (role, dens, scorer) in {
        "leadA": (ROLE_LEAD, d_lead, lambda n: score_lead(n, progs["A"], root_pc, mode)),
        "leadB": (ROLE_LEAD, d_lead, lambda n: score_lead(n, progs["B"], root_pc, mode)),
        "bassA": (ROLE_BASS, d_bass, lambda n: score_bass(n, progs["A"], root_pc, mode)),
        "bassB": (ROLE_BASS, d_bass, lambda n: score_bass(n, progs["B"], root_pc, mode)),
        "drums": (ROLE_DRUM, d_drum, score_drums),
    }.items():
        # 动机只喂给主旋律轨：模型从动机结束处续写完剩余小节
        prompt = spec.get("motif") if key in ("leadA", "leadB") else None
        cands = gacha(gen, role, n_cand, seed, tags, dens, prompt=prompt,
                      quiet=quiet)
        picks[key] = pick(cands, scorer, key, quiet=quiet)

    if picks["leadA"] is None:
        raise SystemExit("模型没有产出可用素材，换一个 seed 再试（--seed）")

    secs = make_sections(n_seg)
    u = 60.0 / bpm / 4.0
    if style == "nes":
        total, layers, ch = _build_nes(picks, progs, secs, u, root_pc, mode, density)
    else:
        total, layers, ch = _build_pop(picks, progs, secs, u, root_pc, mode, density)
    return total, layers, picks, ch, u


def _build_nes(picks, progs, secs, u, root_pc, mode, density):
    """芯片编排：脉冲1 主旋律 / 脉冲2 琶音 / 三角波低音 / 噪声鼓"""
    ch = {"lead": [], "harm": [], "bass": [], "drums": []}
    total = 0
    for si, (name, pkey, lead_key, use_arp, use_bass, use_drums, vk, _oct) \
            in enumerate(secs):
        base = si * STEPS
        total = base + STEPS
        prog = progs[pkey]

        if use_arp:
            for bar in range(4):
                _, _, pcs = chord_of(root_pc, mode, prog, bar)
                tones = chord_tones(pcs, 60)
                pat = [tones[0], tones[1], tones[2], tones[0] + 12,
                       tones[2], tones[1], tones[0] + 12, tones[2]]
                for k in range(16):
                    st = base + bar * 16 + k
                    p = pat[k % 8]
                    v = (0.62 if k % 4 == 0 else 0.44) * vk
                    ch["harm"].append((st * u, (st + 1) * u, midi_hz(p),
                                       min(v, 1.0)))
        if use_bass:
            bass = picks["bassA"] if pkey == "A" else picks["bassB"]
            for s, e, p in bass or []:
                d = max(e - s, 1)
                ch["bass"].append(((base + s) * u, (base + s + d * 0.92) * u,
                                   midi_hz(p), 0.80 * vk))
        if use_drums:
            for s, _e, p in prep_drums(picks["drums"]):
                v = 0.85 if p in (36, 35) else (0.62 if p in (38, 40) else 0.34)
                ch["drums"].append(((base + s) * u, 0.0, float(p),
                                    min(v * vk, 1.0)))
        if lead_key:
            lead = picks["leadA"] if lead_key == "A" else picks["leadB"]
            oct_up = 12 if name.endswith("3") else 0
            for s, e, p in lead or []:
                pp = p + oct_up
                if pp > 96:
                    pp -= 12
                d = max(e - s, 1)
                gap = 0.5 if d > 1 else 0.15          # 芯片味的短断奏
                ch["lead"].append(((base + s) * u,
                                   (base + s + max(d - gap, 0.5)) * u,
                                   midi_hz(pp), 0.78 * vk))
    return total, nes_layers_to_midi_layers(ch), ch


def _build_pop(picks, progs, secs, u, root_pc, mode, density):
    """抒情编排：钢琴主旋律 / 铺底 / 尼龙吉他分解 / 贝斯 / 鼓"""
    layers = []

    def emit(prog, name, is_drum=False):
        layers.append([name, prog, name, is_drum, []])
        return layers[-1][4]

    e_lead = emit(P_PIANO, "melody")
    e_pad = emit(P_PAD, "warm pad")
    e_str = emit(P_STR, "strings")
    e_arp = emit(P_GTR, "nylon guitar")
    e_bass = emit(P_BASS_POP, "bass")
    e_drum = emit(0, "drums", is_drum=True)

    total = 0
    for si, (_name, pkey, lead_key, use_arp, use_bass, use_drums, vk, _oct) \
            in enumerate(secs):
        base = si * STEPS
        total = base + STEPS
        prog = progs[pkey]

        def V(v, s):
            pos = int(s) % 16
            acc = 1.08 if pos == 0 else (1.00 if pos % 4 == 0 else 0.90)
            return int(max(1, min(127, v * vk * acc)))

        for bar in range(4):
            _, _, pcs = chord_of(root_pc, mode, prog, bar)
            s0, s1 = (base + bar * 16) * u, (base + bar * 16 + 16) * u
            for p in chord_tones(pcs, 57):
                e_pad.append((V(58, bar * 16), s0, s1, p))
                e_str.append((V(44, bar * 16), s0, s1, p))
        if use_arp:
            for bar in range(4):
                _, _, pcs = chord_of(root_pc, mode, prog, bar)
                tones = chord_tones(pcs, 60)
                pat = [tones[0], tones[1], tones[2], tones[0] + 12,
                       tones[2], tones[1]]
                for k in range(8):
                    st = base + bar * 16 + k * 2
                    e_arp.append((V(66 if k % 2 == 0 else 54, st),
                                  st * u, (st + 2) * u, pat[k % 6]))
        if use_bass:
            bass = picks["bassA"] if pkey == "A" else picks["bassB"]
            for s, e, p in bass or []:
                e_bass.append((V(96, s), (base + s) * u,
                               (base + s + max(e - s, 1) * 0.95) * u, p))
        if use_drums:
            for s, _e, p in prep_drums(picks["drums"]):
                v = 92 if p in (36, 35) else (78 if p in (38, 40) else 48)
                e_drum.append((V(v, s), (base + s) * u,
                               (base + s + 1) * u, int(p)))
        if lead_key:
            lead = picks["leadA"] if lead_key == "A" else picks["leadB"]
            for s, e, p in lead or []:
                e_lead.append((V(100, s), (base + s) * u,
                               (base + s + max(e - s, 1) * 0.96) * u, p))

    end, hold = total, 8
    for p in chord_tones(chord_of(root_pc, mode, progs["A"], 0)[2], 57):
        e_str.append((92, end * u, (end + hold) * u, p))
        e_pad.append((70, end * u, (end + hold) * u, p))
    e_bass.append((96, end * u, (end + hold) * u, 33 + (root_pc - 9) % 12))
    e_drum.append((104, end * u, (end + 4) * u, 49))
    total = end + hold
    return total, layers, None


def nes_layers_to_midi_layers(ch):
    """芯片渲染用的 (t0,t1,freq,vel01) -> MIDI layers"""
    out = []

    def back(vals, prog, name, is_drum=False, use_pitch=False):
        notes = []
        for t0, t1, f, v in vals:
            if use_pitch:
                p, t1e = int(round(f)), (t1 if t1 > t0 else t0 + 0.05)
            else:
                p, t1e = int(round(PITCH_A4 + 12 * math.log2(f / 440.0))), t1
            notes.append((int(v * 127), t0, max(t1e, t0 + 0.04), p))
        out.append([name, prog, name, is_drum, notes])

    back(ch["lead"], P_SQUARE, "pulse1 lead")
    back(ch["harm"], P_SQUARE, "pulse2 arp")
    back(ch["bass"], P_TRI, "triangle bass")
    back(ch["drums"], 0, "noise drums", is_drum=True, use_pitch=True)
    return out


def midi_hz(p):
    return 440.0 * (2.0 ** ((float(p) - PITCH_A4) / 12.0))


# --------------------------------------------------------------------------- #
def write_midi(path, layers, bpm):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    for _stem, prog, name, is_drum, notes in layers:
        ins = pretty_midi.Instrument(program=prog, name=name, is_drum=is_drum)
        for v, s, e, p in notes:
            ins.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p),
                                              start=max(0.0, float(s)),
                                              end=max(0.02, float(e))))
        if ins.notes:
            pm.instruments.append(ins)
    pm.write(path)
    return pm.get_end_time(), sum(len(x[4]) for x in layers)
