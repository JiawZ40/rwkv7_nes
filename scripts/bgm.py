# -*- coding: utf-8 -*-
"""bgm.py —— Skill 的唯一入口：一段描述 -> 一个 MIDI 工程 + 一个 WAV

    python scripts/bgm.py --text "地下洞窟，神秘，中速，32 小节"
    python scripts/bgm.py --spec spec.json          # LLM 解析后的结构化请求
    python scripts/bgm.py --schema                  # 打印 spec 规范给 LLM 看

产出两个文件：
    <out>/<name>.mid   多轨标准 MIDI（可直接拖进 DAW 改音符 / 换音色 / 导出）
    <out>/<name>.wav   2A03 芯片音源或 FluidSynth 渲染的成品

最后往 stdout 打一行 JSON，方便调用方（LLM / 脚本）直接解析。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

import compose as C          # noqa: E402
import intent as I           # noqa: E402
import postfx as PF          # noqa: E402
import render_wav as R       # noqa: E402

NOTE_LABEL = {"C": "C", "Cs": "C#", "D": "D", "Ds": "D#", "E": "E", "F": "F",
              "Fs": "F#", "G": "G", "Gs": "G#", "A": "A", "As": "A#", "B": "B"}


def build_spec(a):
    if a.spec:
        spec = I.load_spec(a.spec)
    elif a.text:
        spec = I.from_text(a.text)
        if a.scene:
            spec["scene"] = a.scene
    else:
        spec = {"scene": a.scene or I.DEFAULT_PRESET}
    for k in ("style", "sound", "mix", "mood", "speed", "density", "mode", "root"):
        v = getattr(a, k, None)
        if v:
            spec[k] = v
    for k in ("bpm", "bars", "seed", "n_cand"):
        v = getattr(a, k, None)
        if v:
            spec[k] = v
    if a.name:
        spec["name"] = a.name
    return I.norm_spec(spec)


def main():
    ap = argparse.ArgumentParser(
        description="RWKV-7 NES BGM skill：文字 -> MIDI + WAV")
    ap.add_argument("--text", default="", help="自然语言描述（规则兜底解析）")
    ap.add_argument("--spec", default="", help="spec JSON 的文件路径或直接是 JSON")
    ap.add_argument("--scene", default="", help="场景名（见 --schema）")
    ap.add_argument("--name", default="", help="输出文件名")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--style", choices=["nes", "pop"])
    ap.add_argument("--sound", choices=["2a03", "sf2"])
    ap.add_argument("--mix", choices=["chip", "wide", "full"],
                    help="2A03 的混音档位：chip=原味干声 / wide=轻加宽 / "
                         "full=加倍+合唱+饱和+总线压缩混响（默认）")
    ap.add_argument("--postfx", metavar="MID", default="",
                    help="只对一份已有 MIDI 重新混音，不跑模型")
    ap.add_argument("--sf2", default=os.environ.get("NES_BGM_SF2", ""),
                    help="FluidSynth 音色库路径（--sound sf2 时用）")
    ap.add_argument("--mood", choices=["heroic", "epic_tension", "calm", "mystery"])
    ap.add_argument("--speed", choices=["slow", "medium", "fast"])
    ap.add_argument("--density", choices=["low", "medium", "high"])
    ap.add_argument("--mode", choices=["minor", "major"])
    ap.add_argument("--root", default="")
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--bars", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-cand", type=int, help="每个声部抽几张卡（默认 3）")
    ap.add_argument("--ckpt", default=os.environ.get("NES_BGM_CKPT", ""))
    ap.add_argument("--no-wav", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="复用上次的抽卡结果（只重渲染，秒出）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--schema", action="store_true", help="打印 spec 规范后退出")
    a = ap.parse_args()

    if a.schema:
        print(I.dump_schema())
        return 0

    # 只做后期：对一份已有 MIDI 重新混音（不跑模型，秒级）
    if a.postfx:
        stems, total = PF.midi_to_stems(a.postfx)
        os.makedirs(os.path.abspath(a.out), exist_ok=True)
        dst = os.path.join(os.path.abspath(a.out),
                           (a.name or os.path.splitext(
                               os.path.basename(a.postfx))[0]) + ".wav")
        PF.enhance_mix(stems, total, dst, preset=a.mix or "full")
        print(json.dumps({"mid": a.postfx, "wav": dst,
                          "mix": a.mix or "full"},
                         ensure_ascii=False))
        return 0

    spec = build_spec(a)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    mid_path = os.path.join(out_dir, spec["name"] + ".mid")
    wav_path = os.path.join(out_dir, spec["name"] + ".wav")
    pick_path = os.path.join(out_dir, spec["name"] + ".picks.json")

    t0 = time.time()
    if not a.quiet:
        print("[spec ] %s | %s %s | %s / %s / %s | bpm=%d bars=%d seed=%d" % (
            spec["name"], NOTE_LABEL[C.NOTE_NAMES[spec["root_pc"]]],
            spec["mode"], spec["mood"], spec["speed"], spec["density"],
            spec["bpm"], spec["bars"], spec["seed"]))

    # ---------- 抽卡 + 编排 ---------- #
    picks = None
    if a.reuse and os.path.exists(pick_path):
        with open(pick_path, encoding="utf-8") as f:
            picks = {k: [tuple(x) for x in v] if v else None
                     for k, v in json.load(f).items()}
        if not a.quiet:
            print("[cache] 复用抽卡结果", pick_path)

    gen = None
    if picks is None:
        from infer import load_generator
        gen, _vocab = load_generator(a.ckpt or None)
        total, layers, picks, ch, u = C.compose(spec, gen, quiet=a.quiet)
        with open(pick_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in picks.items()}, f)
    else:
        total, layers, ch, u = replay(spec, picks)

    # ---------- MIDI ---------- #
    seconds, n_notes = C.write_midi(mid_path, layers, spec["bpm"])
    detail = " ".join("%s=%d" % (x[2].replace(" ", ""), len(x[4]))
                      for x in layers if x[4])
    if not a.quiet:
        print("[midi ] %s  %.1fs  %s" % (mid_path, seconds, detail))

    # ---------- WAV ---------- #
    wav = None
    if not a.no_wav:
        if spec["sound"] == "sf2":
            R.render_sf2(mid_path, wav_path, a.sf2 or None, quiet=a.quiet)
        else:
            mix = spec.get("mix", "full")
            if mix == "chip":
                # 原味：单声道拼起来的干冷芯片音（FC 直出）
                if ch is None:
                    ch = R.layers_to_chiptune(layers)
                R.render_chiptune(ch, total * u + 0.8, wav_path, quiet=a.quiet)
            else:
                stems = ch if ch is not None else PF.layers_to_stems(layers)
                PF.enhance_mix(stems, max(seconds, total * u) + 0.8, wav_path,
                               preset=mix, quiet=a.quiet)
        wav = wav_path

    result = {
        "mid": mid_path,
        "wav": wav,
        "seconds": round(seconds, 1),
        "bpm": spec["bpm"],
        "key": "%s %s" % (NOTE_LABEL[C.NOTE_NAMES[spec["root_pc"]]], spec["mode"]),
        "scene": spec.get("scene", ""),
        "style": spec["style"],
        "sound": spec["sound"],
        "mix": spec.get("mix", "full"),
        "tracks": len([x for x in layers if x[4]]),
        "notes": n_notes,
        "seed": spec["seed"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def replay(spec, picks):
    """用缓存的抽卡结果重建编排（不再跑模型）"""
    style = spec["style"]
    root_pc, mode = spec["root_pc"], spec["mode"]
    n_seg = max(1, spec["bars"] // 4)
    secs = C.make_sections(n_seg)
    u = 60.0 / spec["bpm"] / 4.0
    progs = C.progs_of(spec)
    if style == "nes":
        total, layers, ch = C._build_nes(picks, progs, secs, u, root_pc, mode,
                                         spec["density"])
    else:
        total, layers, ch = C._build_pop(picks, progs, secs, u, root_pc, mode,
                                         spec["density"])
    return total, layers, ch, u


if __name__ == "__main__":
    sys.exit(main())
