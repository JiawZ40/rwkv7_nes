# -*- coding: utf-8 -*-
"""render_wav.py —— MIDI 工程 -> 可听的 WAV

两种音源：

  2a03  自研的 FC（Nintendo 2A03）合成器：2 路脉冲 + 三角波 + 1bit 噪声，
        4bit 量化 + 两级低通模拟电视喇叭。这才是真正的「红白机音色」，
        不依赖任何音色库，离线可用。
  sf2   调用 FluidSynth 渲染 MIDI（需要系统装 fluidsynth 且给一个 .sf2），
        适合把同一份 MIDI 工程换成别的音色试听。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 44100


def lowpass_fft(x, sr, fc):
    """频域一阶低通（不依赖 scipy）"""
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    H = 1.0 / (1.0 + 1j * f / float(fc))
    return np.fft.irfft(np.fft.rfft(x) * H, n).astype(np.float32)


def np_pad(x, n):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def render_chiptune(ch, total_sec, path, quiet=False):
    """ch = {"lead"/"harm"/"bass": [(t0,t1,freq,vel01)], "drums": [(t0,_,gm,vel)]}"""
    import nes_synth as ns

    if not quiet:
        print("[2a03] 渲染 %.1fs ..." % total_sec)
    lead = ns.render_tonal(ch["lead"], "pulse", duty=0.5, total=total_sec,
                           gain=0.26, vib=0.004)
    harm = ns.render_tonal(ch["harm"], "pulse", duty=0.25, total=total_sec,
                           gain=0.19)
    bass = ns.render_tonal(ch["bass"], "triangle", total=total_sec, gain=0.40)
    drum = ns.render_drums(ch["drums"], total=total_sec, gain=0.30)

    # 谐波一直铺到 18kHz 会发刺；FC 实际是经电视小喇叭低通播出的
    lead = lowpass_fft(lead, SR, 9000)
    harm = lowpass_fft(harm, SR, 7500)
    bass = lowpass_fft(bass, SR, 4000)
    drum = lowpass_fft(drum, SR, 7000)

    n = max(len(lead), len(harm), len(bass), len(drum))
    L = (np_pad(lead, n) * 1.00 + np_pad(harm, n) * 0.62 +
         np_pad(bass, n) * 0.95 + np_pad(drum, n) * 0.85)
    R = (np_pad(lead, n) * 0.88 + np_pad(harm, n) * 1.00 +
         np_pad(bass, n) * 0.95 + np_pad(drum, n) * 0.90)
    L = ns.soft_clip(L, 1.15)
    R = ns.soft_clip(R, 1.15)
    L = lowpass_fft(lowpass_fft(L, SR, 9000), SR, 9000)
    R = lowpass_fft(lowpass_fft(R, SR, 9000), SR, 9000)
    fo = int(0.6 * SR)
    L[-fo:] *= np.linspace(1, 0, fo)
    R[-fo:] *= np.linspace(1, 0, fo)
    ns.save_wav(path, L, R, peak=0.90)
    return path


def render_sf2(midi_path, wav_path, sf2=None, quiet=False):
    """用 FluidSynth 把 MIDI 渲染成 WAV（需要 fluidsynth 在 PATH 里）"""
    exe = shutil.which("fluidsynth")
    if not exe:
        raise SystemExit("没找到 fluidsynth，请安装后重试，或用 --sound 2a03")
    if not sf2 or not os.path.exists(sf2):
        raise SystemExit("需要一个 .sf2 音色库（--sf2 路径）")
    cmd = [exe, "-ni", "-F", wav_path, "-r", str(SR), sf2, midi_path]
    if not quiet:
        print("[sf2 ]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return wav_path


def layers_to_chiptune(layers):
    """把任意 MIDI layers 折进 2A03 的四条通道（脉冲/脉冲/三角/噪声）

    FC 同时只能发 5 个音（2 脉冲 + 三角 + 噪声 + DPCM），所以「铺底」
    这类声部会被并到脉冲 2 上，听感反而更接近真机。
    """
    ch = {"lead": [], "harm": [], "bass": [], "drums": []}
    mapping = {"pulse1 lead": "lead", "melody": "lead",
               "pulse2 arp": "harm", "nylon guitar": "harm",
               "warm pad": "harm", "strings": "harm",
               "triangle bass": "bass", "bass": "bass",
               "noise drums": "drums", "drums": "drums"}
    for _stem, _prog, name, is_drum, notes in layers:
        key = mapping.get(name, "lead")
        for v, t0, t1, p in notes:
            if is_drum:
                ch["drums"].append((t0, 0.0, float(p), v / 127.0))
            else:
                ch[key].append((t0, t1, 440.0 * 2 ** ((p - 69.0) / 12.0),
                                v / 127.0))
    for k in ch:
        ch[k].sort()
    return ch


def wav_peak(path):
    import wave
    with wave.open(path) as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2")
    return float(np.max(np.abs(data))) / 32768.0 if n else 0.0
