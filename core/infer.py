# -*- coding: utf-8 -*-
"""RWKV-7 四小节循环生成器 —— 纯 PyTorch 推理内核（不需要 CUDA / triton）

这个文件是从 src/generate.py 里抽出来的推理部分：
  * Theory         把控制标签（TONE/MOOD/INST）翻译成解码期的乐理先验
  * LoopGenerator  状态机约束下的因果采样，产出 [(start, end, pitch), ...]

只依赖 core/rwkv7_torch.py（纯 torch 重写，参数名与 fla 一致，可直接加载
训练权重），不再需要 fla / triton / transformers，所以任何装了 CPU 版 torch
的机器都能跑。串行 CPU 推理一条 4 小节素材约 3-5 秒。

对外只有一个入口：load_generator(ckpt) -> (LoopGenerator, Vocab)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import (Vocab, D_MINOR_PCS, parse_tone, scale_pcs,          # noqa: E402
                   INST_RANGES, INST_POLY, GM_DRUM_CORE, GM_DRUM_OTHER)

# --------------------------------------------------------------------------- #
ROOT_D = 2
D_HARM = D_MINOR_PCS | {11}          # + B natural (melodic-minor raised 6th)
BRASS_LO, BRASS_HI = 43, 84          # horn/trombone section tessitura (default)
MAX_POLY = 4                         # section-style voice count (default)
STRONG = {0, 4, 8, 12}               # strong beats in 4/4 (16th grid)
STEPS = 64                           # 4 bars x 16

MINOR_DEG = (0, 2, 3, 5, 7, 8, 10)   # scale degree index -> semitone interval
MAJOR_DEG = (0, 2, 4, 5, 7, 9, 11)

# 情绪 -> 音级（按「音阶级数」而非半音，故换调式时自动正确）
MOOD_DEGREE_W = {
    "MOOD_EPIC_TENSION": {0: 1.35, 1: 0.85, 2: 1.20, 3: 1.25, 4: 1.30, 5: 1.10, 6: 0.70},
    "MOOD_HEROIC":       {0: 1.40, 1: 0.90, 2: 1.25, 3: 1.15, 4: 1.35, 5: 1.05, 6: 0.85},
    "MOOD_CALM":         {0: 1.25, 1: 1.10, 2: 1.10, 3: 1.15, 4: 1.20, 5: 0.90, 6: 0.90},
    "MOOD_MYSTERY":      {0: 1.20, 1: 1.15, 2: 1.05, 3: 0.90, 4: 1.00, 5: 1.20, 6: 1.05},
}
DEFAULT_DEGREE_W = {0: 1.30, 2: 1.15, 3: 1.20, 4: 1.25, 6: 0.80}


class Theory:
    """把控制标签翻译成解码时的乐理先验

    原来是硬编码的 D 小调 + 铜管音域，换标签也不会变，导致「输入 prompt」
    形同虚设（换调性后先验仍按 D 小调加权）。这里按标签推导：
      * TONE_<Root>_<MODE> -> 主音 / 调式 -> 调内音级集合与音级权重
      * MOOD_*             -> 音级倾向（史诗偏根音/五度，神秘偏 b2/b6…）
      * INST_*             -> 音域（tessitura）与最大复音数
    """

    def __init__(self, tags=()):
        tags = tuple(tags)
        self.mood = next((t for t in tags if t.startswith("MOOD_")),
                         "MOOD_EPIC_TENSION")
        self.tone = next((t for t in tags if t.startswith("TONE_")),
                         "TONE_D_MINOR")
        self.inst = next((t for t in tags if t.startswith("INST_")),
                         "INST_BRASS_SECTION")
        self.root_pc, self.mode = parse_tone(self.tone)
        self.deg = MINOR_DEG if self.mode == "MINOR" else MAJOR_DEG
        self.pcs = scale_pcs(self.root_pc, self.mode)
        # 小调额外允许旋律小调的升六级（D 小调即 B 自然）
        self.allowed = set(self.pcs)
        if self.mode == "MINOR":
            self.allowed.add((self.root_pc + 9) % 12)
        self.lo, self.hi = INST_RANGES.get(self.inst, (BRASS_LO, BRASS_HI))
        self.max_poly = INST_POLY.get(self.inst, MAX_POLY)
        self.w = MOOD_DEGREE_W.get(self.mood, DEFAULT_DEGREE_W)
        # 打击乐用 GM 打击乐映射，音高不是音高而是乐器编号 ——
        # 调式音级 / 强拍锚音对它毫无意义，必须走单独分支，
        # 否则会把「底鼓在强拍」学成「D 小调的主音在强拍」。
        self.is_perc = self.inst == "INST_NOISE_PERC"

    # ---- pitch -> log weight ---- #
    def bias(self, pitch, beat):
        if self.is_perc:
            return self._bias_perc(pitch, beat)
        iv = (pitch - self.root_pc) % 12
        deg = self.deg.index(iv) if iv in self.deg else -1
        if pitch % 12 in self.allowed and deg >= 0:
            w = self.w.get(deg, 0.9)
        elif pitch % 12 in self.allowed:
            w = 0.9                      # 调内但音级表未列出（如小调升六级）
        else:
            w = 0.35                     # 调外音强压
        if not (self.lo <= pitch <= self.hi):
            w *= 0.12                    # 出音域
        if beat in STRONG:
            if deg in (0, 4):            # 根音 / 五度锚音
                w *= 1.6
            elif deg in (2, 3, 5):       # 三度 / 四度 / 六度
                w *= 1.3
        if pitch < self.lo - 12 or pitch > self.hi + 12:
            w *= 0.05                    # 离音域太远
        return math.log(max(w, 1e-3))

    def _bias_perc(self, pitch, beat):
        """GM 打击乐映射：音高 = 乐器编号，不参与调式约束"""
        if not (self.lo <= pitch <= self.hi):
            return math.log(1e-3)        # 35-81 之外根本不是鼓
        w = GM_DRUM_CORE.get(pitch, GM_DRUM_OTHER)
        # 鼓的骨架在强拍：底鼓/军鼓落 1、3 拍，踩镲铺满
        if beat in STRONG:
            w *= 1.25 if pitch in (35, 36, 38, 40) else 1.05
        else:
            w *= 1.15 if pitch in (42, 44, 46) else 1.0   # 弱拍靠踩镲填充
        return math.log(max(w, 1e-3))


def pitch_log_bias(pitch, beat):
    """向后兼容：默认（D 小调 / 铜管）先验"""
    return _DEFAULT_THEORY.bias(pitch, beat)


_DEFAULT_THEORY = Theory(("MOOD_EPIC_TENSION", "TONE_D_MINOR",
                          "DENSITY_HIGH", "INST_BRASS_SECTION"))


# --------------------------------------------------------------------------- #
class LoopGenerator:
    """state-machine constrained causal sampling (incremental recurrent cache)"""

    def __init__(self, model, vocab: Vocab, device):
        self.model, self.v, self.device = model, vocab, device
        # 纯 torch 后端支持增量推理（每步只算 1 个 token）；fla 后端只能整段重算
        self.incremental = hasattr(model, "prefill") and hasattr(model, "decode")

    # ---------------------------------------------------------------- #
    # music-prompt (动机) handling
    # ---------------------------------------------------------------- #
    @staticmethod
    def sanitize_prompt(notes, th, max_prompt_step=None):
        """把用户动机规整到 16 分网格：夹紧步数、限制复音、丢弃非法音符

        动机必须给模型留出续写空间，所以默认只允许占前 STEPS-4 步
        （最后 4 个 16 分位留给收尾，保证循环能闭合）。
        """
        lim = STEPS - 4 if max_prompt_step is None else \
            max(1, min(int(max_prompt_step), STEPS - 4))
        cands = []
        for n in notes or []:
            try:
                s, e, p = int(n[0]), int(n[1]), int(n[2])
            except Exception:
                continue
            if e <= s or p < 0 or p > 127 or s >= lim:
                continue
            cands.append((max(0, s), min(e, STEPS), p))
        cands.sort()
        keep, live_spans = [], []
        for s, e, p in cands:
            if sum(1 for (a, b) in live_spans if a <= s < b) >= th.max_poly:
                continue                       # 超过该乐器的复音上限
            keep.append((s, e, p))
            live_spans.append((s, e))
        return sorted(keep)

    def encode_prompt(self, notes, th, max_prompt_step=None):
        """动机 -> 前缀 token（BAR / POS / OFF / ON）

        返回 (ids, events, active, cur_step, bars_seen, on_count)，
        与 generate() 内部的状态变量一一对应，便于无缝续写。
        """
        v = self.v
        notes = self.sanitize_prompt(notes, th, max_prompt_step)
        if not notes:
            return [], [], set(), -1, -1, 0

        ids, events = [], []
        active, on_count = set(), 0
        cur_step, bars_seen = -1, -1

        # 动机必须是「开头连续的一小段」：遇到空小节就截断
        # （语法不允许连续两个 BAR，跳过空小节会造出非法 token 序列）
        bars = []
        for bar in range(4):
            b0 = bar * 16
            starts = sorted({s - b0 for s, _, _ in notes if b0 <= s < b0 + 16})
            ends = sorted({e - b0 for _, e, _ in notes
                           if b0 <= e < b0 + 16 and e < STEPS})
            if not starts and not ends:
                break
            bars.append((bar, b0, sorted(set(starts) | set(ends))))

        for bar, b0, positions in bars:
            ids.append(v.bar_id)
            bars_seen += 1
            for pos in positions:
                st = b0 + pos
                ids.append(v.pos_ids[pos])
                cur_step = st
                for _, e, p in sorted(notes, key=lambda n: n[2]):
                    if e == st and p in active:
                        ids.append(v.off_ids[p])
                        events.append((st, "OFF", p))
                        active.discard(p)
                for s, _, p in sorted(notes, key=lambda n: n[2]):
                    if s == st:
                        ids.append(v.on_ids[p])
                        events.append((st, "ON", p))
                        active.add(p)
                        on_count += 1
        return ids, events, active, cur_step, bars_seen, on_count

    @torch.no_grad()
    def generate(self, tags, temperature=1.0, top_p=0.96, density_target=10,
                 seed=None, max_tok=300, prompt_notes=None, max_prompt_step=None):
        """因果采样生成一段 4 小节循环

        tags          控制标签 prompt（MOOD/TONE/SPEED/DENSITY/INST）
        prompt_notes  音乐动机 prompt：[(start, end, pitch)]（16 分音符步），
                      会被编码成 BAR/POS/ON/OFF 前缀 token 强制喂给模型，
                      模型从动机结束处续写完剩余小节，并在 step 64 闭合循环。
        """
        v = self.v
        rng = np.random.default_rng(seed)
        th = Theory(tags)

        # ---- 动机 prompt 编码为前缀 token ---- #
        prompt_ids, p_events, p_active, p_step, p_bars, p_on = \
            self.encode_prompt(prompt_notes, th, max_prompt_step=max_prompt_step)

        ids = [v.bos_id] + [v.stoi[t] for t in tags] + prompt_ids

        active = set(p_active)
        events = list(p_events)         # (step, kind, pitch)
        cur_step = p_step
        bars_seen = p_bars
        on_count = p_on

        # ---- prefix pass: BOS + control tags ---- #
        # NOTE: fla's recurrent Cache is incompatible with transformers 5.17 on
        # this stack, so we re-run the prefix each step.  The input is padded to
        # a FIXED length (pad tokens sit in the causal future, so they cannot
        # affect earlier logits) -- this keeps the triton kernel shape-stable.
        # 纯 torch 后端不需要这些：state 可以带着走，每步只解码 1 个 token。
        FWD = 256
        pad_id = v.pad_id

        def fwd_logits(seq):
            buf = np.full(FWD, pad_id, dtype=np.int64)
            buf[: len(seq)] = seq
            x = torch.from_numpy(buf).unsqueeze(0).to(self.device)
            return self.model(input_ids=x, use_cache=False).logits[0, len(seq) - 1].float().clone()

        gst = None
        if self.incremental:
            logits, gst = self.model.prefill(ids, device=self.device)
            logits = logits.float()
        else:
            logits = fwd_logits(ids)

        def add(toks):
            allow[torch.tensor(toks, device=self.device)] = True

        for t in range(min(max_tok, FWD - len(ids) - 1)):
            last = v.itos[ids[-1]]
            allow = torch.zeros(len(v), dtype=torch.bool, device=self.device)
            if last == "<BOS>" or last in (v.itos[i] for i in v.ctrl_ids):
                add([v.bar_id])
            elif last == "BAR":
                add(v.pos_ids)                 # any position may open a bar
            elif last.startswith("POS_"):
                # close hanging pitches first, then attacks
                for p in active:
                    allow[v.off_ids[p]] = True
                add(v.on_ids)
            elif last.startswith("OFF_") or last.startswith("ON_"):
                if active:
                    for p in active:
                        allow[v.off_ids[p]] = True
                if len(active) < th.max_poly and cur_step < STEPS - 1:
                    add(v.on_ids)
                if cur_step >= 0 and cur_step + 1 < STEPS:
                    # any later position within this bar may follow
                    nxt_pos = (cur_step % 16) + 1
                    if nxt_pos < 16:
                        add(v.pos_ids[nxt_pos:])
                    else:
                        add([v.bar_id])       # bar line crossed
                if cur_step >= STEPS - 1 and not active:
                    allow[v.eos_id] = True
            logits = logits.masked_fill(~allow, float("-inf"))

            # ---------------------- theory shaping ---------------------- #
            if last.startswith("POS_") or last.startswith("OFF_") or last.startswith("ON_"):
                beat = cur_step % 16 if cur_step >= 0 else 0
                for p in range(128):
                    logits[v.on_ids[p]] += th.bias(p, beat)
                # density schedule pressure
                should = density_target * 4 * (cur_step + 1) / STEPS
                if on_count > should + 2:
                    logits[torch.tensor(v.on_ids, device=self.device)] -= 0.8
                elif on_count < should - 2:
                    logits[torch.tensor(v.on_ids, device=self.device)] += 0.35
                # seam suppression: no new attacks in the final two 16ths
                if cur_step >= STEPS - 2:
                    logits[torch.tensor(v.on_ids, device=self.device)] -= 3.0
                # encourage closing long-held notes at the seam
                if cur_step >= STEPS - 4 and active:
                    for p in active:
                        logits[v.off_ids[p]] += 0.8

            # ---------------------- sample (top-p) ---------------------- #
            probs = torch.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, -1)
            keep = cum - sp < top_p
            keep[0] = True
            sp = sp * keep
            sp = sp / sp.sum()
            nxt = int(si[int(torch.multinomial(sp, 1))])
            ids.append(nxt)
            tok = v.itos[nxt]

            # ---------------------- track state ---------------------- #
            if tok == "BAR":
                bars_seen += 1
            elif tok.startswith("POS_"):
                cur_step = bars_seen * 16 + int(tok[4:])
            elif tok.startswith("OFF_"):
                p = int(tok[4:])
                active.discard(p)
                if cur_step >= 0:
                    events.append((cur_step, "OFF", p))
            elif tok.startswith("ON_"):
                p = int(tok[3:])
                active.add(p)
                on_count += 1
                if cur_step >= 0:
                    events.append((cur_step, "ON", p))
            elif tok == "<EOS>":
                break

            if cur_step >= STEPS - 1 and not active:
                break

            # ---- next-step forward ---- #
            if len(ids) < FWD:
                if gst is not None:
                    logits, gst = self.model.decode(nxt, gst, device=self.device)
                    logits = logits.float()
                else:
                    logits = fwd_logits(ids)

        # -------- loop-align closure: force-close hanging notes on the seam ------- #
        notes, held = [], {}
        for st, kind, p in sorted(events, key=lambda e: (e[0], e[1] == "ON")):
            if st >= STEPS:
                continue
            if kind == "ON":
                held[p] = st
            else:
                s0 = held.pop(p, None)
                if s0 is not None and st > s0:
                    notes.append((s0, st, p))
        for p, s0 in held.items():          # close exactly at step 64 == next loop's 0
            notes.append((s0, STEPS, p))
        notes.sort()
        return ids, notes


# --------------------------------------------------------------------------- #
# 模型加载：只用纯 torch 后端
# --------------------------------------------------------------------------- #
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_CKPT = os.path.join(MODEL_DIR, "rwkv7_midi.pt")
DEFAULT_CFG = dict(d_model=384, n_layer=6, n_head=6)


def load_generator(ckpt=None, device=None, **cfg):
    """加载权重，返回 (LoopGenerator, Vocab)

    ckpt   权重路径，默认 models/rwkv7_midi.pt（安装时由 download_model.py 拉取）
    device 默认自动（有 CUDA 用 CUDA，否则 CPU）
    """
    from rwkv7_torch import RWKV7Cfg, load_torch_model

    ckpt = ckpt or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        raise SystemExit(
            "找不到模型权重：%s\n请先运行：python scripts/download_model.py" % ckpt)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    conf = dict(DEFAULT_CFG)
    conf.update(cfg)
    vocab = Vocab()
    rcfg = RWKV7Cfg(vocab_size=len(vocab), hidden_size=conf["d_model"],
                    num_hidden_layers=conf["n_layer"], num_heads=conf["n_head"])
    model = load_torch_model(ckpt, rcfg, device=device, dtype=torch.float32)
    return LoopGenerator(model, vocab, device), vocab
