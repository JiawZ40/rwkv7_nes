# nes-bgm — NES / Famicom Style Game BGM Generator (AI Skill)

A 12M-parameter RWKV-7 small model that runs entirely on your CPU, purpose-built for composing game music in the NES/Famicom style. It does **not** "generate an audio clip" — it **writes sheet music**: producing an editable multi-track MIDI project file plus a finished WAV render.

- Input: a single natural-language sentence ("underground cave, mysterious, medium tempo") or a structured spec
- Output: `<name>.mid` (4 tracks, named parts, directly editable in any DAW) + `<name>.wav` (2A03 chip audio)
- Environment: pure CPU, offline, no API calls, no GPU required; ~40–70 seconds per 32-bar generation

## Installation

```bash
bash install.sh                    # macOS / Linux / Git-Bash / Termux
powershell -ExecutionPolicy Bypass -File install.ps1   # Windows
```

Common options:

| Flag | Purpose |
|---|---|
| `--target codex` / `workbuddy` / `claude` / `cursor` / `local` | Choose which skills directory to install into (default: auto-detect) |
| `--from <repo_path>` | Use a local checkout; skip network |
| `--no-model` | Install code only; download weights yourself later |

After installation, the script prints `SKILL_DIR` — the absolute path to this directory.

## Usage

```bash
# 1) Fastest: one-liner (keyword fallback parsing)
python scripts/bgm.py --text "overworld adventure, heroic, fast" --bars 32 --out ./bgm_out

# 2) Recommended: let the host LLM produce a spec (better semantic understanding; see SKILL.md)
python scripts/bgm.py --spec spec.json --out ./bgm_out

# 4) Re-mix only, without re-running the model (seconds)
python scripts/bgm.py --text "underground cave, mysterious, medium" --reuse --mix full --out ./bgm_out
python scripts/bgm.py --postfx some.mid --mix full --out ./bgm_out   # re-mix an existing MIDI

# 5) Print the spec schema (for LLM reference)
python scripts/bgm.py --schema
```

Common switches: `--bpm --bars --seed --n-cand --style nes|pop --sound 2a03|sf2
--mix chip|wide|full --root --mode --mood --speed --density`

## Mix Modes `--mix`

The raw 2A03 output is dry, cold mono stitched together (L/R correlation 0.99). `scripts/postfx.py` applies an enhanced mix on the **same synthesizer** without loading any external sound bank:

| Mode | What it does | Best for |
|---|---|---|
| `full` (default) | Lead double-tracked (±7 cents + 9/14 ms Haas), harmonies chorused (±4 cents + 3 ms LFO chorus), bass tanh saturation + octave-down sub, bus-wide reverb → soft compression → air shelf → soft limiter | Wanting a "finished record" feel |
| `wide` | Same but tamer (±5/±3 cents, less reverb) | Wanting thickness while retaining chip character |
| `chip` | No enhancement; raw Famicom output | True vintage dry sound |

Measured on 32 bars: L/R correlation 0.99 → 0.86–0.91, overall RMS +3~4 dB, >8 kHz energy
0.1% → 0.6% (brightened without harshness), peak 0.94 with zero clipping; rendering overhead +8~13 seconds.

## Directory Structure

```
nes-bgm/
├── SKILL.md              Behavioral spec for the host LLM (required reading)
├── README.md             This file
├── install.sh / install.ps1
├── requirements.txt
├── core/                 Self-contained inference kernel (no dependency on the rest of the repo)
│   ├── vocab.py          Vocabulary and control-label space
│   ├── rwkv7_torch.py    Pure-PyTorch RWKV-7 (prefill/decode incremental inference)
│   └── infer.py          Theory music priors + LoopGenerator constrained sampling
├── scripts/
│   ├── bgm.py            Sole entry point: description -> MIDI + WAV
│   ├── intent.py         Intent parsing (scene preset table + text fallback)
│   ├── compose.py        Candidate sampling + music scoring + multi-track arrangement
│   ├── compose.py        Candidate sampling + music scoring + multi-track arrangement
│   ├── nes_synth.py      2A03 synthesizer (pulse/triangle/noise)
│   ├── render_wav.py     MIDI -> WAV (2A03 original flavor or FluidSynth)
│   ├── postfx.py         Enhanced mixing pipeline (doubling/chorus/saturation/bus compression & reverb)
│   └── download_model.py Weight download (GitHub / Gitee / mirror multi-source)
└── models/rwkv7_midi.pt  Weights (24 MB, downloaded at install time)
```

## How It Works

1. **Intent → control labels**: A scene vocabulary translates "underground cave" into `MOOD_MYSTERY / TONE_A_MINOR / SPEED_MEDIUM / DENSITY_MEDIUM`, or the host LLM provides a spec directly.
2. **Small model writes motifs (sampling)**: For each voice, N four-bar candidates are generated and the best one is picked by music scoring (chord fit ×3, stepwise smoothness, rhythmic stability, register range).
3. **Deterministic rules do arrangement**: Harmony progression, arpeggio texture, instrumentation, and dynamics are hard-coded, so voices always agree — the model only handles the "write the motif" step.
4. **Per-track export + chip rendering**: Four channels are synthesized through 2A03 pulse/triangle/noise and passed through two stages of low-pass to emulate a TV speaker.

## Weights

`models/rwkv7_midi.pt` (24 MB, identical to `ckpt/best.pt`).

```bash
python scripts/download_model.py                       # auto-pick a working mirror
python scripts/download_model.py --source gitee        # prefer domestic (China) mirror
python scripts/download_model.py --from ../ckpt/best.pt  # use a local training checkpoint
```
