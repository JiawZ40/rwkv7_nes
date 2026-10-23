---
name: nes-bgm
description: 用本地 RWKV-7 小模型（12M，CPU 可跑）写 NES / FC 红白机风格的游戏 BGM，并同时产出可编辑的多轨 MIDI 工程文件和成品 WAV。当用户要给游戏做 8-bit / 芯片 / 红白机风格配乐，或明确要求「要 MIDI 工程、要能改音符」而不是只要一段音频时使用。
---

# NES 风格游戏 BGM 生成器

一个装在你们机器上的小模型（RWKV-7，12M 参数，纯 PyTorch CPU 推理），
专门写红白机风格的游戏音乐。它不做「音频生成」，而是**写谱**：
输出标准多轨 MIDI（带声部名、分轨、力度），任何 DAW / 在线编辑器都能打开改。

## 什么时候用

- 用户要游戏 BGM：大地图、洞窟、城堡、Boss 战、水下、城镇、标题、通关、Game Over……
- 用户要 8-bit / chip / 红白机 / FC / 复古像素游戏的音乐
- 用户要**能改的曲子**（"把主旋律第 3 小节改一下" "换成钢琴" "导出分轨"）
- 用户要循环播放的游戏循环素材

不要在用户要人声歌曲、管弦乐总谱、或超过 2 分钟的交响结构时用——
模型是在 NES-MDB + 中文流行 MIDI 上训的，强项是短动机 + 循环织体。

## 怎么用（三步）

### 1. 把用户的描述解析成 spec

读用户的话，推断出下面这些字段。**不要照抄用户的原话当参数**，
要理解意图后落进这套受控词汇里（模型只认这些词）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `scene` | 见下方场景表 | 一句话定整套参数，最先填 |
| `root` | `C Cs D Ds E F Fs G Gs A As B` 或 0-11 | 调根音，用户没说就用场景默认 |
| `mode` | `minor` \| `major` | 小调=暗/战斗，大调=明亮/日常 |
| `mood` | `heroic` \| `epic_tension` \| `calm` \| `mystery` | 英勇 / 紧张 / 平静 / 神秘 |
| `speed` | `slow` \| `medium` \| `fast` | 只影响模型倾向，实际速度看 bpm |
| `density` | `low` \| `medium` \| `high` | 音符密度，这个标签对模型最有效 |
| `bpm` | 60-220 | 芯片音乐一般 120-180 |
| `bars` | 4 的倍数 | 32 小节 ≈ 45-60 秒；游戏循环建议 16-32 |
| `style` | `nes`（默认）\| `pop` | 芯片编排 / 抒情编排 |
| `sound` | `2a03`（默认）\| `sf2` | 真 FC 音源 / FluidSynth 音色库 |
| `mix` | `full`（默认）\| `wide` \| `chip` | 混音档位，见下方「后期混音」 |
| `name` | 英文文件名 | 不要用空格和中文 |
| `seed` | 整数 | 同一个 seed 结果完全可复现 |
| `motif` | `[[start,end,pitch],...]` | 可选。用户给动机时让它续写（步 0-63） |

场景表（直接映射成 mood/speed/density/bpm/mode）：

```
overworld 大地图/草原/森林   → heroic, fast,  medium, bpm 168, major
underworld 洞窟/地牢/地下    → mystery, medium, medium, bpm 132, minor
castle 城堡/神殿/遗迹        → epic_tension, fast, high, bpm 150, minor
boss Boss战/最终关/战斗      → epic_tension, fast, high, bpm 176, minor
underwater 水下/海底         → calm, medium, low, bpm 112, major
town 城镇/商店/菜单          → calm, medium, medium, bpm 120, major
victory 胜利/通关/结算       → heroic, fast, high, bpm 160, major
gameover 失败/死亡           → epic_tension, slow, low, bpm 84, minor
title 标题/开场/主菜单       → heroic, medium, medium, bpm 140, minor
shoot 射击/太空/赛博         → epic_tension, fast, high, bpm 184, minor
puzzle 解谜/思考             → mystery, slow, low, bpm 100, minor
credits 片尾/结尾            → calm, slow, low, bpm 92, major (style=pop)
ballad 抒情/钢琴/情歌        → calm, slow, low, bpm 92, minor (style=pop)
```

### 2. 写 spec 文件并调用

```bash
python "$SKILL_DIR/scripts/bgm.py" --spec spec.json --out ./bgm_out
```

`$SKILL_DIR` 是本 skill 的安装目录（如 `~/.codex/skills/nes-bgm`）。
spec.json 示例：

```json
{
  "name": "castle_boss",
  "scene": "boss",
  "root": "D",
  "mode": "minor",
  "mood": "epic_tension",
  "speed": "fast",
  "density": "high",
  "bpm": 168,
  "bars": 32,
  "seed": 7
}
```

不写 spec 也行（关键词兜底，语义理解会差）：

```bash
python "$SKILL_DIR/scripts/bgm.py" --text "地下洞窟，神秘，中速" --bars 32
python "$SKILL_DIR/scripts/bgm.py" --schema        # 打印完整 schema 和场景表
```

对一份**已有的 MIDI** 重新混音（不跑模型，秒级）：

```bash
python "$SKILL_DIR/scripts/bgm.py" --postfx path/to/xxx.mid --mix full --out ./bgm_out
```

### 3. 回报给用户

命令最后会打一行 JSON：`{"mid": "...", "wav": "...", "seconds": 45.2, "key": "D minor", ...}`。

**两个文件都要给用户**：
- `.mid` —— 可编辑的工程（4 轨：pulse1 lead / pulse2 arp / triangle bass / noise drums），
  可以拖进任何 DAW 改音符、换音色、导出分轨。这是本 skill 相对「直接生成音频」的核心价值，
  一定要主动告诉用户「这是能改的谱子，不是死音频」。
- `.wav` —— 2A03 芯片音源渲染的成品，直接听。

## 后期混音（默认开，别乱关）

2A03 直出是单声道拼起来的，干、薄、窄（实测左右相关度 0.99 ≈ 单声道）。
`scripts/postfx.py` 在**同一个合成器**上做混音，不加任何外部音色库：

- `full`（默认）主旋律 double-track（±7 音分 + 9/14ms Haas）、和声合唱化
  （±4 音分 + 3ms LFO chorus）、贝斯 tanh 饱和 + 低八度 sub、
  总线统一混响 → 软压缩 → 空气感 shelf → 软限幅
- `wide` 同上但收敛（失谐 ±5/+3，混响更少），保留更多芯片味
- `chip` 不开增强，FC 直出的干冷原味

实测（32 小节）：左右相关度 0.99 → 0.86-0.91，整体 RMS +3~4dB，
高频（>8kHz）0.1% → 0.6%，峰值 0.94 零削波。渲染耗时约 +8~13 秒。

想要"更复古/更干"就用 `--mix chip`；用户嫌"太满/太润"就退到 `--mix wide`。
只改混音不想重新抽卡时，加 `--reuse`（秒级）。

## 迭代（很重要）

用户说"不好听"/"换个感觉"时，按这个顺序改，**每次只改一个变量**：

1. `--seed 8 9 10` 换抽卡（同一个 spec 换个种子，旋律完全不同）
2. 换 `scene` 或 `mood`
3. 改 `bpm`
4. 改 `density`（对模型的实际影响最大）

已经跑过一次后，加 `--reuse` 可以复用上次的抽卡结果只重渲染，1-2 秒出结果，
适合反复调 bpm / 配器。

## 环境与耗时

- 依赖：`numpy` `torch`(CPU) `pretty_midi`；渲染 2A03 不需要任何音色库。
- 权重 24MB，在 `$SKILL_DIR/models/rwkv7_midi.pt`，缺失时跑
  `python "$SKILL_DIR/scripts/download_model.py"`。
- 一次 32 小节的生成：CPU 串行约 40-70 秒（默认每个声部抽 3 张卡挑最好的，
  `--n-cand 1` 可以缩到 20 秒）。**先告诉用户要等一会儿**，别中途重复调用。
- 完全离线，不联网，不调任何 API。

## 模型能力边界（别吹牛）

模型在 NES-MDB（56%）+ 中文流行 MIDI（44%）上训练，只会写 4 小节一组的短素材：
- 强项：芯片风格动机、琶音、循环素材、单声部旋律续写
- 弱项：电影管弦、爵士、复杂曲式、长线条发展
- `density` 标签对输出影响显著，`mood` 影响很弱（别指望换个 mood 就完全不同）
- 和声、织体、配器是确定性规则写死的，所以**永远不会跑调**；
  模型只负责「写动机」这一步
