# -*- coding: utf-8 -*-
"""nes_synth.py —— 2A03（FC/红白机）风格合成器

语料本身就是 NES，用 GM 音色库渲染会把它变成「通用电子音」，失去芯片味。
这里直接合成 2A03 的三类通道：
  * 2 路脉冲波（Pulse）：占空比 12.5% / 25% / 50%，音量 4bit（16 级）
  * 1 路三角波（Triangle）：低音，无音量包络，只有通断
  * 1 路噪声（Noise）：1bit LFSR 质感，用于鼓组

要点：
  * 加法合成并限制谐波数量到 18kHz 以下，避免混叠（方波不能直接 np.sign）
  * 用「绝对时间」生成相位，保证音头相位连续，不会出现咔哒
  * 通道音量按 4bit 量化 —— 这是 FC 音色的一个真实特征
"""
import numpy as np

SR = 44100
NYQ_LIM = 18000.0


# ------------------------------------------------------------------ #
# 基础波形（加法合成，带限）
# ------------------------------------------------------------------ #
def _harmonics(f, nmax_floor=1):
    n = int(min(80.0, NYQ_LIM / max(f, 1e-6)))
    return np.arange(1, max(n, nmax_floor) + 1, dtype=np.float64)


def pulse(t, f, duty=0.5, vib_depth=0.0, vib_rate=5.5):
    """脉冲波（占空比 duty）。方波 = duty 0.5"""
    n = _harmonics(f)
    a = (4.0 / np.pi) * np.sin(np.pi * n * duty) / n
    ph = 2.0 * np.pi * f * t
    if vib_depth > 0:
        # 颤音：瞬时频率 f(t)=f(1+d·sin(2πfm t))，相位是它的积分，
        # 直接对 sin 乘调制会变成 AM 而不是 FM，这里用正确的积分形式。
        ph = ph - (f * vib_depth / vib_rate) * np.cos(2.0 * np.pi * vib_rate * t)
    out = np.zeros_like(t)
    for i, (nn, aa) in enumerate(zip(n, a)):
        if abs(aa) < 1e-4:
            continue
        out += aa * np.sin(nn * ph)
    return out * 0.5


def triangle(t, f):
    """三角波：只有奇次谐波，幅度按 1/n^2 衰减（比脉冲波更柔）"""
    n = _harmonics(f)
    n = n[n % 2 == 1]
    a = (8.0 / np.pi ** 2) * ((-1.0) ** ((n - 1) // 2)) / (n ** 2)
    ph = 2.0 * np.pi * f * t
    out = np.zeros_like(t)
    for nn, aa in zip(n, a):
        out += aa * np.sin(nn * ph)
    return out


def noise(n, seed):
    """1bit 噪声：LFSR 的本质是 ±1 的随机方波，比高斯白噪更「脆」"""
    rng = np.random.default_rng(seed)
    return np.sign(rng.standard_normal(n))


# ------------------------------------------------------------------ #
# 包络
# ------------------------------------------------------------------ #
def _env(k, sr, atk=0.004, rel=0.012):
    """FC 没有真正的音量包络：直通 + 极短的收放（只为消咔哒）"""
    e = np.ones(k)
    a = max(1, int(atk * sr))
    r = max(1, int(rel * sr))
    if k > a * 2:
        e[:a] = np.linspace(0, 1, a)
    else:
        e[:k // 2] = np.linspace(0, 1, max(1, k // 2))
    if k > r * 2:
        e[-r:] = np.linspace(1, 0, r)
    return e


def _decay(k, sr, tau):
    t = np.arange(k) / sr
    return np.exp(-t / max(tau, 1e-3))


# ------------------------------------------------------------------ #
# 通道渲染
# ------------------------------------------------------------------ #
def render_tonal(notes, kind="pulse", duty=0.5, total=None, sr=SR,
                 gain=0.30, vib=0.0, quant4=True):
    """notes: [(t0, t1, freq, vel01)]"""
    if not notes:
        n = int((total or 1.0) * sr)
        return np.zeros(n)
    n = int((total if total else max(e for _, e, _, _ in notes) + 0.4) * sr)
    buf = np.zeros(n)
    for t0, t1, f, v in notes:
        i0 = int(t0 * sr)
        i1 = min(int(t1 * sr), n)
        if i1 <= i0 or f <= 0:
            continue
        t = (np.arange(i1 - i0) + i0) / sr            # 绝对时间 -> 相位连续
        if kind == "pulse":
            x = pulse(t, f, duty=duty, vib_depth=vib)
        elif kind == "triangle":
            x = triangle(t, f)
        else:                                          # pulse 25%
            x = pulse(t, f, duty=0.25, vib_depth=vib)
        if quant4:                                     # 4bit 音量（16 级）
            v = round(float(v) * 15.0) / 15.0
        buf[i0:i1] += x * _env(i1 - i0, sr) * float(v)
    return buf * gain


# 鼓：GM 音高 -> 音色
def render_drums(notes, total=None, sr=SR, gain=0.34, seed=1234):
    if not notes:
        n = int((total or 1.0) * sr)
        return np.zeros(n)
    n = int((total if total else max(s for s, _, _, _ in notes) + 1.0) * sr)
    buf = np.zeros(n)
    rng = np.random.default_rng(seed)

    def add(i0, x):
        i1 = min(i0 + len(x), n)
        if i1 > i0:
            buf[i0:i1] += x[:i1 - i0]

    for t0, _, p, v in notes:
        p = int(p)
        i0 = int(t0 * sr)
        vel = max(0.05, float(v))
        if p in (35, 36):                              # 底鼓：三角波下滑
            k = int(0.26 * sr)
            t = np.arange(k) / sr
            f = 130.0 * np.exp(-t / 0.045) + 46.0
            ph = 2 * np.pi * np.cumsum(f) / sr
            x = np.sin(ph) * _decay(k, sr, 0.10)
            add(i0, x * vel * 1.0)
        elif p in (38, 40):                            # 军鼓：噪声 + 一点音高
            k = int(0.17 * sr)
            nz = noise(k, rng.integers(1 << 30)) * _decay(k, sr, 0.045)
            t = np.arange(k) / sr
            tone = np.sin(2 * np.pi * 190 * t) * _decay(k, sr, 0.03) * 0.5
            add(i0, (nz * 0.85 + tone) * vel)
        elif p in (42, 44):                            # 闭镲
            k = int(0.06 * sr)
            nz = noise(k, rng.integers(1 << 30)) * _decay(k, sr, 0.012)
            nz = np.diff(nz, prepend=0.0)              # 微分 -> 提亮
            add(i0, nz * vel * 0.55)
        elif p == 46:                                  # 开镲
            k = int(0.30 * sr)
            nz = noise(k, rng.integers(1 << 30)) * _decay(k, sr, 0.10)
            nz = np.diff(nz, prepend=0.0)
            add(i0, nz * vel * 0.5)
        elif p in (49, 57, 55):                        # 吊镲
            k = int(1.3 * sr)
            nz = noise(k, rng.integers(1 << 30)) * _decay(k, sr, 0.45)
            nz = np.diff(nz, prepend=0.0)
            add(i0, nz * vel * 0.42)
        elif p in (41, 43, 45, 47, 48):                # 嗵鼓
            k = int(0.30 * sr)
            t = np.arange(k) / sr
            f = (110 + (p - 41) * 22) * np.exp(-t / 0.10) + 60
            ph = 2 * np.pi * np.cumsum(f) / sr
            x = np.sin(ph) * _decay(k, sr, 0.11)
            add(i0, x * vel * 0.8)
        else:
            k = int(0.08 * sr)
            add(i0, noise(k, rng.integers(1 << 30)) * _decay(k, sr, 0.02)
                * vel * 0.4)
    return buf * gain


# ------------------------------------------------------------------ #
def soft_clip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def save_wav(path, L, R, sr=SR, peak=0.92):
    y = np.stack([L, R])
    m = np.abs(y).max()
    if m > 0:
        y = y * (peak / m)
    pcm = (np.clip(y, -1, 1).T * 32767).astype(np.int16)
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path
