# -*- coding: utf-8 -*-
"""download_model.py —— 拉取 24MB 的 RWKV-7 权重（多源自动切换）

用法：
    python scripts/download_model.py                 # 自动挑一个能下的源
    python scripts/download_model.py --source gitee  # 指定源
    python scripts/download_model.py --from ../ckpt/best.pt   # 用本地已有的权重

源优先级：环境变量 > gitee（国内快） > github > 镜像
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "models", "rwkv7_midi.pt")

# --------------------------------------------------------------------------- #
# 仓库地址：发布时把 OWNER / REPO 换成你自己的，或者用环境变量覆盖
#   export NESBGM_GITHUB=https://github.com/yourname/nes-bgm-rwkv7
#   export NESBGM_GITEE=https://gitee.com/yourname/nes-bgm-rwkv7
# --------------------------------------------------------------------------- #
GH = os.environ.get("NESBGM_GITHUB", "https://github.com/OWNER/nes-bgm-rwkv7")
GT = os.environ.get("NESBGM_GITEE", "https://gitee.com/OWNER/nes-bgm-rwkv7")
TAG = os.environ.get("NESBGM_TAG", "v1.0.0")
FILE = "rwkv7_midi.pt"

SOURCES = [
    ("env", os.environ.get("NES_BGM_MODEL_URL", "")),
    ("gitee", "%s/releases/download/%s/%s" % (GT, TAG, FILE)),
    ("github", "%s/releases/download/%s/%s" % (GH, TAG, FILE)),
    ("hf-mirror", "https://hf-mirror.com/OWNER/nes-bgm-rwkv7/resolve/main/%s" % FILE),
]

EXPECTED_SHA256 = os.environ.get("NES_BGM_SHA256", "")


def _progress(done, total):
    if total and total > 0:
        pct = done * 100.0 / total
        sys.stdout.write("\r  %.1f%%  %.1f/%.1f MB" %
                         (pct, done / 1e6, total / 1e6))
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")


def fetch(url, dst):
    print("[get  ]", url)
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "nes-bgm-skill"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = r.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            _progress(done, total)
    os.replace(tmp, dst)
    return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto",
                    choices=["auto", "github", "gitee", "hf-mirror", "env"])
    ap.add_argument("--from", dest="src", default="",
                    help="直接用本地已存在的权重文件（复制过来）")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    if os.path.exists(MODEL) and not a.force:
        print("[ok   ] 权重已存在", MODEL, "%.1f MB" %
              (os.path.getsize(MODEL) / 1e6))
        return 0

    if a.src:
        if not os.path.exists(a.src):
            raise SystemExit("找不到权重文件：" + a.src)
        shutil.copyfile(a.src, MODEL)
        print("[copy ] %s -> %s" % (a.src, MODEL))
    else:
        order = SOURCES if a.source == "auto" else \
            [s for s in SOURCES if s[0] == a.source]
        last = None
        for name, url in order:
            if not url:
                continue
            try:
                fetch(url, MODEL)
                break
            except Exception as e:                       # noqa: BLE001
                last = e
                print("  [fail] %s: %s" % (name, e))
        else:
            raise SystemExit(
                "所有源都下载失败（最后一个错误：%s）\n"
                "可以手动下载权重放到 %s，或用 "
                "python scripts/download_model.py --from <本地权重路径>" % (last, MODEL))

    size = os.path.getsize(MODEL)
    if size < 1e6:
        raise SystemExit("下载的文件太小（%.1f KB），可能不是权重" % (size / 1e3))
    print("[ok   ] %s  %.1f MB" % (MODEL, size / 1e6))
    if EXPECTED_SHA256:
        got = sha256(MODEL)
        print("[sha  ]", got)
        if got != EXPECTED_SHA256:
            raise SystemExit("sha256 不匹配，文件可能损坏")
    return 0


if __name__ == "__main__":
    sys.exit(main())
