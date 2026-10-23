# nes-bgm —— NES / FC 红白机风格游戏 BGM 生成器（AI Skill）

一个 12M 参数的 RWKV-7 小模型，专门写红白机风格的游戏音乐，不生成音频，而是**写谱**：产出可编辑的多轨 MIDI 工程 + 成品 WAV。

- 输入：一句自然语言（"地下洞窟，神秘，中速"）或一份结构化 spec
- 输出：`<name>.mid`（4 轨，带声部名，DAW 可直接改）+ `<name>.wav`（2A03 芯片音源）
- 环境：纯 CPU、离线、无 API、无显卡要求；一次 32 小节约 40-70 秒

## 安装

```bash
bash install.sh                    # macOS / Linux / Git-Bash / Termux
powershell -ExecutionPolicy Bypass -File install.ps1   # Windows
```

常用参数：

| 参数                                                             | 作用                       |
| -------------------------------------------------------------- | ------------------------ |
| `--target codex` / `workbuddy` / `claude` / `cursor` / `local` | 指定装到哪个 skills 目录（默认自动探测） |
| `--from <仓库路径>`                                                | 用本地已有仓库，不联网              |
| `--no-model`                                                   | 只装代码，权重之后自己下             |

装完会打印 `SKILL_DIR`，也就是本目录的绝对路径。

## 用法

```bash
# 1) 最快：一句话（关键词兜底解析）
python scripts/bgm.py --text "大地图冒险，热血，快" --bars 32 --out ./bgm_out

# 2) 推荐：让宿主 LLM 产出 spec（语义理解更好，见 SKILL.md）
python scripts/bgm.py --spec spec.json --out ./bgm_out

# 4) 只改混音、不重跑模型（秒级）
python scripts/bgm.py --text "地下洞窟，神秘，中速" --reuse --mix full --out ./bgm_out
python scripts/bgm.py --postfx some.mid --mix full --out ./bgm_out   # 已有 MIDI 重新混音

# 5) 打印 spec 规范（给 LLM 看）
python scripts/bgm.py --schema
```

常用开关：`--bpm --bars --seed --n-cand --style nes|pop --sound 2a03|sf2 --mix chip|wide|full --root --mode --mood --speed --density`

## 混音档位 `--mix`

2A03 直出是干冷的单声道拼接（左右相关度 0.99）。`scripts/postfx.py` 在**同一个
合成器**上做增强混音，不加任何外部音色库：

| 档位         | 做法                                                                                                                      | 适合         |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| `full`（默认） | 主旋律 double-track（±7 音分 + 9/14ms Haas）、和声合唱化（±4 音分 + 3ms LFO chorus）、贝斯 tanh 饱和 + 低八度 sub、总线统一混响 → 软压缩 → 空气感 shelf → 软限幅 | 想要"成品感"    |
| `wide`     | 同上但收敛（±5/+3 音分，混响更少）                                                                                                    | 想要厚度但保留芯片味 |
| `chip`     | 不增强，FC 直出                                                                                                               | 想要真正的复古干声  |

实测 32 小节：左右相关度 0.99 → 0.86-0.91，整体 RMS +3\~4dB，>8kHz 能量
0.1% → 0.6%（提亮但不刺），峰值 0.94 零削波；渲染耗时 +8\~13 秒。

## 目录结构

```
nes-bgm/
├── SKILL.md              给宿主 LLM 读的行为规范（必读）
├── README.md             本文件
├── install.sh / install.ps1
├── requirements.txt
├── core/                 自包含推理内核（不依赖仓库其它部分）
│   ├── vocab.py          词表与控制标签空间
│   ├── rwkv7_torch.py    纯 PyTorch 的 RWKV-7（prefill/decode 增量推理）
│   └── infer.py          Theory 乐理先验 + LoopGenerator 受限采样
├── scripts/
│   ├── bgm.py            唯一入口：描述 -> MIDI + WAV
│   ├── intent.py         意图解析（场景预设表 + 文本兜底）
│   ├── compose.py        抽卡 + 乐理评分 + 多轨编排
│   ├── compose.py        抽卡 + 乐理评分 + 多轨编排
│   ├── nes_synth.py      2A03 合成器（脉冲/三角/噪声）
│   ├── render_wav.py     MIDI -> WAV（2A03 原味 或 FluidSynth）
│   ├── postfx.py         增强混音管线（加倍/合唱/饱和/总线压缩混响）
│   └── download_model.py 权重下载（GitHub / Gitee / 镜像多源）
└── models/rwkv7_midi.pt  权重（24MB，安装时下载）
```

## 它是怎么工作的

1. **意图 → 控制标签**：场景词表把「地下洞窟」翻译成 `MOOD_MYSTERY / TONE_A_MINOR / SPEED_MEDIUM / DENSITY_MEDIUM`，或由宿主 LLM 直接给 spec。
2. **小模型写动机（抽卡）**：每个声部生成 N 个 4 小节候选，按乐理评分（和弦契合 ×3、级进平滑、节奏稳定、音域宽度）挑最好的一张。
3. **确定性规则做编曲**：和声进行、琶音织体、配器、力度是写死的，所以跨声部永远协和——模型只负责「写动机」这一步。
4. **分轨导出 + 芯片渲染**：4 条通道分别用 2A03 的脉冲/三角/噪声合成，再过两级低通模拟电视喇叭。

## 权重

`models/rwkv7_midi.pt`（24MB，与 `ckpt/best.pt` 相同）。

```bash
python scripts/download_model.py                       # 自动挑一个能下的源
python scripts/download_model.py --source gitee        # 国内优先
python scripts/download_model.py --from ../ckpt/best.pt  # 用本地训练产物
```

