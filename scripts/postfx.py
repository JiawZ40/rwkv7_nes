# -*- coding: utf-8 -*-
"""postfx.py —— 增强混音管线（把「干冷的芯片音」混成「能听的成品」）

2A03 原味（`render_wav.render_chiptune`）是单声道拼起来的，干、薄、窄。
这个管线在**同一个 2A03 合成器**的基础上做混音，不加任何外部音色库：

  * 主旋律   double-track：±7 音分失谐 + 9/14ms Haas + 左右铺开
  * 和声/琶音 合唱化：±4 音分 + 3ms LFO chorus + 立体声拉开
  * 贝斯     tanh 饱和激励 + 低八度 sub（130Hz 以下）+ 低频 shelf
  * 鼓       去隆隆 + 亮度提升，瞬态不压
  * 总线     统一梳齿混响 → 软压缩 → 空气感 shelf → 软限幅 → 归一化

只依赖 numpy，完全离线。

用法：
    python scripts/postfx.py in.mid out.wav          # 对已有 MIDI 重新混音
    python scripts/postfx.py in.mid out.wav --preset chip
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nes_synth as ns          # noqa: E402

SR = 44100


# --------------------------------------------------------------------------- #
# 滤波（FFT 域，不依赖 scipy）
# --------------------------------------------------------------------------- #
def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(2, n))))


def _filt(x, kind, fc, gain_db=0.0, high=True):
    """FFT 域滤波。

    零填充到 2 的幂再算 —— numpy 对非 2 幂长度（这里 260 万点，含 5^n 因子）
    的 FFT 会慢一个数量级，实测渲染时间从 55s 降到 6s。
    """
    n = len(x)
    if n == 0:
        return x
    N = _next_pow2(n + 8192)
    f = np.fft.rfftfreq(N, 1.0 / SR)
    if kind == "lp":
        H = 1.0 / (1.0 + 1j * f / float(fc))
    elif kind == "hp":
        w = f / float(fc)
        H = 1j * w / (1.0 + 1j * w)
    else:                                            # shelf
        g = 10.0 ** (gain_db / 20.0)
        r = (f / fc) ** 2 if high else (fc / np.maximum(f, 1e-6)) ** 2
        H = 1.0 + (g - 1.0) * r / (1.0 + r)
    return np.fft.irfft(np.fft.rfft(x, N) * H, N)[:n].astype(np.float64)


def fft_lp(x, fc):
    return _filt(x, "lp", fc)


def fft_hp(x, fc):
    return _filt(x, "hp", fc)


def air_shelf(x, fc=3200.0, gain_db=2.6):
    """高频空气感：从 fc 起缓慢抬升到 gain_db"""
    return _filt(x, "shelf", fc, gain_db, high=True)


def low_shelf(x, fc=150.0, gain_db=3.0):
    return _filt(x, "shelf", fc, gain_db, high=False)


def _conv(x, h):
    N = _next_pow2(len(x) + len(h))
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N),
                        N)[:len(x) + len(h)]


# --------------------------------------------------------------------------- #
# 混响：指数衰减噪声 IR + 早期反射（比梳齿网络快一个数量级）
# --------------------------------------------------------------------------- #
from functools import lru_cache          # noqa: E402


@lru_cache(maxsize=8)
def make_ir(tail=1.4, tone=5200.0, seed=7):
    n = int(tail * SR)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / (tail / 3.5))
    for dl, g in ((0.011, 0.55), (0.019, 0.42), (0.029, 0.34), (0.037, 0.27)):
        ir[int(dl * SR)] += g                      # 早期反射给出房间尺寸感
    ir = fft_lp(ir, tone)
    ir = fft_hp(ir, 180.0)
    return ir


def reverb(x, amount=0.30, tail=1.4, tone=5200.0):
    """wet 的 RMS 对齐到 amount × dry RMS，语义比固定增益可控"""
    if amount <= 0 or len(x) == 0:
        return np.zeros(len(x))
    wet = _conv(x, make_ir(tail, tone))
    rx = np.sqrt(np.mean(x ** 2) + 1e-12)
    rw = np.sqrt(np.mean(wet ** 2) + 1e-12)
    return wet * (amount * rx / rw)


# --------------------------------------------------------------------------- #
# 动态
# --------------------------------------------------------------------------- #
def compress(x, thresh=0.32, ratio=2.4, atk_ms=8, rel_ms=140, makeup=1.0):
    """分块 RMS 前馈压缩（无 lookahead，够用且稳定）"""
    win = max(1, int(0.005 * SR))
    n = len(x) // win
    if n < 2:
        return x * makeup
    rms = np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    env = np.zeros(n)
    ka = np.exp(-1.0 / (0.001 * atk_ms * SR / win))
    kr = np.exp(-1.0 / (0.001 * rel_ms * SR / win))
    e = 0.0
    for i in range(n):
        e = rms[i] if rms[i] > e else kr * e + (1 - kr) * rms[i]
        env[i] = ka * e + (1 - ka) * env[i - 1] if i else e
    gain = np.where(env > thresh,
                    (thresh + (env - thresh) / ratio) / np.maximum(env, 1e-9),
                    1.0)
    g = np.repeat(gain, win)
    if len(g) < len(x):
        g = np.pad(g, (0, len(x) - len(g)), constant_values=g[-1])
    return x * g[:len(x)] * makeup


def chorus(x, depth_ms=3.0, rate=0.7):
    """LFO 调制的微延迟 —— 合唱感（比纯失谐更像「多个人在弹」）"""
    n = len(x)
    if n < 8:
        return x
    t = np.arange(n) / SR
    d = 0.001 * depth_ms * SR * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    idx = np.clip(np.arange(n) - d, 0, n - 1)
    return np.interp(idx, np.arange(n), x)


# --------------------------------------------------------------------------- #
# 分轨渲染
# --------------------------------------------------------------------------- #
def render_stem(notes, kind, duty=0.5, total=1.0, gain=0.3, vib=0.0,
                lp=9000.0, detunes=(0.0,), pan=(0.5,), delays=(0.0,),
                chorus_ms=0.0):
    """返回 [(mono, pan01, delay_sec), ...]，每个失谐副本一份"""
    outs = []
    for dt, pn, dl in zip(detunes, pan, delays):
        shifted = [(t0 + dl, t1 + dl, f * (2.0 ** (dt / 1200.0)), v)
                   for t0, t1, f, v in notes]
        mono = ns.render_tonal(shifted, kind, duty=duty, total=total,
                               gain=gain, vib=vib, quant4=(dt == 0.0))
        mono = fft_lp(mono, lp)
        if chorus_ms > 0:
            mono = chorus(mono, depth_ms=chorus_ms, rate=0.55 + 0.3 * dt)
        outs.append((mono, pn, dl))
    return outs


def render_drums_stem(notes, total=1.0, gain=0.34, lp=8500.0, air=1.6):
    mono = ns.render_drums(notes, total=total, gain=gain)
    mono = fft_hp(mono, 55.0)                     # 去隆隆声
    mono = fft_lp(mono, lp)
    if air:
        mono = air_shelf(mono, 5200.0, air)
    return [(mono, 0.5, 0.0)]


def bass_thicken(notes, total=1.0, gain=0.46, lp=5200.0, sat=1.8,
                 sub_gain=0.55):
    """贝斯：饱和激励 + 低八度 sub + 低频 shelf"""
    mono = ns.render_tonal(notes, "triangle", total=total, gain=gain)
    mono = fft_lp(mono, lp)
    drive = np.tanh(mono * sat) / np.tanh(sat)
    low = ns.render_tonal([(t0, t1, f * 0.5, v * 0.9) for t0, t1, f, v in notes],
                          "triangle", total=total, gain=gain * 0.9)
    low = fft_lp(low, 130.0)                      # 只留 sub
    return [(0.85 * mono + 0.45 * drive + sub_gain * low, 0.5, 0.0)]


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
PRESETS = {
    "full": dict(
        lead=dict(gain=0.30, lp=11500.0, detunes=(0.0, 7.0, -7.0),
                  pan=(0.50, 0.24, 0.76), delays=(0.0, 0.009, 0.014), wet=0.20),
        harm=dict(gain=0.21, lp=9000.0, detunes=(0.0, 4.0, -4.0),
                  pan=(0.32, 0.12, 0.88), delays=(0.0, 0.011, 0.016),
                  chorus_ms=3.0, wet=0.30),
        bass=dict(gain=0.46, lp=5200.0, sat=1.8, sub_gain=0.55, wet=0.04),
        drums=dict(gain=0.36, lp=8500.0, air=1.6, wet=0.16),
        bus=dict(compress=dict(thresh=0.32, ratio=2.4), air=(3000.0, 2.4),
                 drive=1.25, clip=1.3, lp=14000.0, peak=0.94),
    ),
    # 轻度增强：保留更多芯片味，只做空间和厚度
    "wide": dict(
        lead=dict(gain=0.28, lp=10000.0, detunes=(0.0, 5.0, -5.0),
                  pan=(0.50, 0.34, 0.66), delays=(0.0, 0.007, 0.011), wet=0.12),
        harm=dict(gain=0.21, lp=8200.0, detunes=(0.0, 3.0, -3.0),
                  pan=(0.38, 0.22, 0.78), delays=(0.0, 0.009, 0.013),
                  chorus_ms=1.5, wet=0.18),
        bass=dict(gain=0.44, lp=4600.0, sat=1.3, sub_gain=0.35, wet=0.03),
        drums=dict(gain=0.35, lp=7600.0, air=0.8, wet=0.10),
        bus=dict(compress=dict(thresh=0.40, ratio=1.8), air=(3600.0, 1.4),
                 drive=1.15, clip=1.3, lp=12000.0, peak=0.92),
    ),
    # 原味：不做加倍/合唱，只走统一混响 + 轻压缩（接近 render_chiptune）
    "chip": dict(
        lead=dict(gain=0.28, lp=9500.0, detunes=(0.0,), pan=(0.5,),
                  delays=(0.0,), wet=0.08),
        harm=dict(gain=0.19, lp=7500.0, detunes=(0.0,), pan=(0.5,),
                  delays=(0.0,), chorus_ms=0.0, wet=0.10),
        bass=dict(gain=0.42, lp=4000.0, sat=1.0, sub_gain=0.0, wet=0.02),
        drums=dict(gain=0.32, lp=7000.0, air=0.0, wet=0.08),
        bus=dict(compress=dict(thresh=0.45, ratio=1.5), air=(4200.0, 0.8),
                 drive=1.10, clip=1.25, lp=11000.0, peak=0.90),
    ),
}
DEFAULT_PRESET = "full"


# --------------------------------------------------------------------------- #
# 主混音
# --------------------------------------------------------------------------- #
def np_pad(x, n):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _pan_gains(p):
    """pan 0=左 1=右 -> 等功率声像"""
    return np.cos(p * np.pi / 2), np.sin(p * np.pi / 2)


def enhance_mix(stems, total_sec, path, preset=DEFAULT_PRESET, quiet=False):
    """stems = {lead/harm/bass: [(t0,t1,freq,vel01)], drums: [(t0,_,gm,vel01)]}"""
    P = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    n = int(total_sec * SR) + int(1.5 * SR)

    L = np.zeros(n)
    R = np.zeros(n)
    wetL = np.zeros(n)
    wetR = np.zeros(n)

    def add(parts, wet):
        nonlocal L, R, wetL, wetR
        mono_sum = np.zeros(n)
        for mono, pan, dl in parts:
            m = np_pad(mono, n)
            gl, gr = _pan_gains(pan)
            mono_sum += m
            i0 = int(dl * SR)
            if dl > 0 and i0 < n:                  # Haas：只有延迟副本才额外铺
                d = np.zeros(n)
                d[i0:] = m[:n - i0]
                L += d * gl * 0.6
                R += d * gr * 0.6
            L += m * gl
            R += m * gr
        if wet > 0 and mono_sum.any():
            w = np_pad(reverb(mono_sum, wet), n)
            wetL += w * 0.85                       # 湿声左右轻微不对称 -> 更宽
            wetR += w * 1.15

    lead = P["lead"]
    add(render_stem(stems.get("lead", []), "pulse", duty=0.5, total=total_sec,
                    gain=lead["gain"], vib=0.004, lp=lead["lp"],
                    detunes=lead["detunes"], pan=lead["pan"],
                    delays=lead["delays"]), lead["wet"])

    harm = P["harm"]
    add(render_stem(stems.get("harm", []), "pulse", duty=0.25, total=total_sec,
                    gain=harm["gain"], lp=harm["lp"], detunes=harm["detunes"],
                    pan=harm["pan"], delays=harm["delays"],
                    chorus_ms=harm.get("chorus_ms", 0.0)), harm["wet"])

    b = P["bass"]
    add(bass_thicken(stems.get("bass", []), total=total_sec, gain=b["gain"],
                     lp=b["lp"], sat=b["sat"], sub_gain=b["sub_gain"]), b["wet"])

    d = P["drums"]
    add(render_drums_stem(stems.get("drums", []), total=total_sec,
                          gain=d["gain"], lp=d["lp"], air=d["air"]), d["wet"])

    # ---------------- 总线 ---------------- #
    bus = P["bus"]
    L, R = L + wetL, R + wetR
    cp = bus["compress"]
    L = compress(L, **cp)
    R = compress(R, **cp)
    fc, gdb = bus["air"]
    L = air_shelf(L, fc, gdb)
    R = air_shelf(R, fc, gdb)
    L = ns.soft_clip(L * bus["drive"], bus["clip"])
    R = ns.soft_clip(R * bus["drive"], bus["clip"])
    L = fft_lp(L, bus["lp"])
    R = fft_lp(R, bus["lp"])

    fo = int(0.6 * SR)
    L[-fo:] *= np.linspace(1, 0, fo)
    R[-fo:] *= np.linspace(1, 0, fo)
    ns.save_wav(path, L, R, peak=bus["peak"])
    if not quiet:
        print("[postfx] preset=%s -> %s" % (preset, path))
    return path


# --------------------------------------------------------------------------- #
# 入口：MIDI / layers -> stems
# --------------------------------------------------------------------------- #
ROLE_MAP = {
    "pulse1 lead": "lead", "melody": "lead",
    "pulse2 arp": "harm", "nylon guitar": "harm",
    "warm pad": "harm", "strings": "harm",
    "triangle bass": "bass", "bass": "bass",
    "noise drums": "drums", "drums": "drums",
}


def midi_to_stems(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    stems = {"lead": [], "harm": [], "bass": [], "drums": []}
    total = pm.get_end_time() + 1.0
    for inst in pm.instruments:
        role = ROLE_MAP.get(inst.name.strip().lower(), "lead")
        if inst.is_drum or role == "drums":
            for nt in inst.notes:
                stems["drums"].append((nt.start, 0.0, float(nt.pitch),
                                       nt.velocity / 127.0))
        else:
            for nt in inst.notes:
                f = 440.0 * 2.0 ** ((nt.pitch - 69.0) / 12.0)
                stems[role].append((nt.start, nt.end, f, nt.velocity / 127.0))
    for k in stems:
        stems[k].sort()
    return stems, total


def layers_to_stems(layers):
    """MIDI layers（bgm.py 的中间产物）-> stems（freq）"""
    stems = {"lead": [], "harm": [], "bass": [], "drums": []}
    for _stem, _prog, name, is_drum, notes in layers:
        role = ROLE_MAP.get(name.strip().lower(), "lead")
        for v, t0, t1, p in notes:
            if is_drum or role == "drums":
                stems["drums"].append((float(t0), 0.0, float(p), v / 127.0))
            else:
                stems[role].append((float(t0), float(t1),
                                    440.0 * 2.0 ** ((p - 69.0) / 12.0),
                                    v / 127.0))
    for k in stems:
        stems[k].sort()
    return stems


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mid")
    ap.add_argument("wav")
    ap.add_argument("--preset", default=DEFAULT_PRESET,
                    choices=sorted(PRESETS.keys()))
    a = ap.parse_args()
    stems, total = midi_to_stems(a.mid)
    enhance_mix(stems, total, a.wav, preset=a.preset)
    print("ok ->", a.wav)
