#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# nes-bgm skill 一键安装（macOS / Linux / Windows Git-Bash / Termux）
#
#   bash install.sh                      # 自动找 skills 目录
#   bash install.sh --target codex       # 装到 ~/.codex/skills
#   bash install.sh --target workbuddy   # 装到 ~/.workbuddy/skills
#   bash install.sh --target local       # 装到当前项目 .workbuddy/skills
#   bash install.sh --from /path/to/repo # 用本地已有的仓库，不联网
#
# 环境变量可覆盖仓库地址（发布时改这里或传参即可）：
#   NESBGM_GITHUB / NESBGM_GITEE / NESBGM_TAG / PIP_INDEX_URL
# --------------------------------------------------------------------------- #
set -euo pipefail

GITHUB_REPO="${NESBGM_GITHUB:-https://github.com/OWNER/nes-bgm-rwkv7}"
GITEE_REPO="${NESBGM_GITEE:-https://gitee.com/OWNER/nes-bgm-rwkv7}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../.." && pwd)"

TARGET="auto"
FROM=""
NO_MODEL=0
PY="${PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2;;
    --from)   FROM="$2"; shift 2;;
    --no-model) NO_MODEL=1; shift;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

say() { printf '\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }

# --------------------------------------------------------------------------- #
# 1. 选一个 python
# --------------------------------------------------------------------------- #
if [[ -z "$PY" ]]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[[ -z "$PY" ]] && { echo "没找到 python，请先安装 Python 3.9+"; exit 1; }
say "python" "$PY ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"

# --------------------------------------------------------------------------- #
# 2. 选一个 skills 目录
# --------------------------------------------------------------------------- #
skills_dir() {
  case "$1" in
    codex)     echo "$HOME/.codex/skills";;
    workbuddy) echo "$HOME/.workbuddy/skills";;
    claude)    echo "$HOME/.claude/skills";;
    cursor)    echo "$HOME/.cursor/skills";;
    local)     echo "$PWD/.workbuddy/skills";;
    auto)
      for d in "$HOME/.workbuddy/skills" "$HOME/.codex/skills" \
               "$HOME/.claude/skills" "$HOME/.cursor/skills"; do
        [[ -d "$d" ]] && { echo "$d"; return; }
      done
      echo "$HOME/.workbuddy/skills";;
    *) echo "$1";;
  esac
}
DEST="$(skills_dir "$TARGET")/nes-bgm"
say "target" "$DEST"

# --------------------------------------------------------------------------- #
# 3. 拿到 skill 源码
# --------------------------------------------------------------------------- #
SRC=""
if [[ -n "$FROM" ]]; then
  SRC="$FROM/skill/nes-bgm"
  [[ -d "$SRC" ]] || { echo "--from 里没有 skill/nes-bgm：$FROM"; exit 1; }
elif [[ -d "$REPO_ROOT/skill/nes-bgm" ]]; then
  SRC="$REPO_ROOT/skill/nes-bgm"          # 已经在本地仓库里执行
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  ok=0
  for url in "$GITEE_REPO" "$GITHUB_REPO"; do
    say "clone" "$url"
    if git clone --depth 1 "$url" "$TMP/repo" >/dev/null 2>&1; then ok=1; break; fi
  done
  [[ $ok -eq 1 ]] || { echo "下载源码失败，检查网络或改用 --from <本地仓库>"; exit 1; }
  SRC="$TMP/repo/skill/nes-bgm"
fi
[[ -d "$SRC" ]] || { echo "找不到 skill 源码：$SRC"; exit 1; }

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
say "install" "skill -> $DEST"

# --------------------------------------------------------------------------- #
# 4. 依赖
# --------------------------------------------------------------------------- #
PIP_ARGS=()
[[ -n "${PIP_INDEX_URL:-}" ]] && PIP_ARGS+=("-i" "$PIP_INDEX_URL")
say "pip" "numpy / pretty_midi ..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install -r "$DEST/requirements.txt" "${PIP_ARGS[@]}"

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  say "pip" "torch (CPU 版，约 200MB，稍等)"
  if ! "$PY" -m pip install torch --index-url \
       "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" 2>/dev/null; then
    say "pip" "CPU 源失败，改用默认源"
    "$PY" -m pip install torch "${PIP_ARGS[@]}"
  fi
else
  say "pip" "torch 已存在，跳过"
fi

# --------------------------------------------------------------------------- #
# 5. 权重
# --------------------------------------------------------------------------- #
if [[ $NO_MODEL -eq 1 ]]; then
  say "model" "跳过（--no-model），用前请跑 scripts/download_model.py"
else
  "$PY" "$DEST/scripts/download_model.py" || \
    say "model" "下载失败，可手动把权重放到 $DEST/models/rwkv7_midi.pt"
fi

# --------------------------------------------------------------------------- #
# 6. 冒烟
# --------------------------------------------------------------------------- #
say "check" "依赖自检"
"$PY" -c "import numpy, torch, pretty_midi; print('  numpy', numpy.__version__, '| torch', torch.__version__, '| pretty_midi ok')"
"$PY" "$DEST/scripts/bgm.py" --schema >/dev/null && say "check" "skill 可调用"

printf '\n\033[1;32m安装完成\033[0m  SKILL_DIR = %s\n\n' "$DEST"
printf '试一句：\n  %s %s/scripts/bgm.py --text "地下洞窟，神秘，中速" --bars 32 --out ./bgm_out\n' "$PY" "$DEST"
printf '产出 ./bgm_out/<name>.mid（可编辑的多轨工程）+ .wav（2A03 芯片音源）\n'
