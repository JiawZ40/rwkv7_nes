# --------------------------------------------------------------------------- #
# nes-bgm skill 一键安装（Windows PowerShell）
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Target codex
#   powershell -ExecutionPolicy Bypass -File install.ps1 -From C:\path\to\repo
#
# 环境变量可覆盖仓库地址：NESBGM_GITHUB / NESBGM_GITEE / NESBGM_TAG
# --------------------------------------------------------------------------- #
param(
    [string]$Target = "auto",          # auto | codex | workbuddy | claude | cursor | local
    [string]$From = "",                # 本地仓库路径（不联网）
    [switch]$NoModel
)

$ErrorActionPreference = "Stop"

$GitHubRepo = if ($env:NESBGM_GITHUB) { $env:NESBGM_GITHUB } else { "https://github.com/OWNER/nes-bgm-rwkv7" }
$GiteeRepo  = if ($env:NESBGM_GITEE)  { $env:NESBGM_GITEE }  else { "https://gitee.com/OWNER/nes-bgm-rwkv7" }
$SelfDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $SelfDir)

function Say($tag, $msg) { Write-Host "[$tag] $msg" -ForegroundColor Cyan }

# ---- 1. python ---- #
$py = $null
foreach ($c in @("python", "python3", "py")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source; break }
}
if (-not $py) { Write-Host "没找到 Python，请先安装 Python 3.9+" -ForegroundColor Red; exit 1 }
Say "python" "$py"

# ---- 2. skills 目录 ---- #
function SkillsDir($t) {
    switch ($t) {
        "codex"     { return (Join-Path $HOME ".codex\skills") }
        "workbuddy" { return (Join-Path $HOME ".workbuddy\skills") }
        "claude"    { return (Join-Path $HOME ".claude\skills") }
        "cursor"    { return (Join-Path $HOME ".cursor\skills") }
        "local"     { return (Join-Path (Get-Location) ".workbuddy\skills") }
        "auto" {
            foreach ($d in @("$HOME\.workbuddy\skills", "$HOME\.codex\skills",
                             "$HOME\.claude\skills", "$HOME\.cursor\skills")) {
                if (Test-Path $d) { return $d }
            }
            return (Join-Path $HOME ".workbuddy\skills")
        }
        default { return $t }
    }
}
$dest = Join-Path (SkillsDir $Target) "nes-bgm"
Say "target" $dest

# ---- 3. 源码 ---- #
if ($From -ne "") {
    $src = Join-Path $From "skill\nes-bgm"
    if (-not (Test-Path $src)) { Write-Host "$From 里没有 skill\nes-bgm" -ForegroundColor Red; exit 1 }
} elseif (Test-Path (Join-Path $RepoRoot "skill\nes-bgm")) {
    $src = Join-Path $RepoRoot "skill\nes-bgm"
} else {
    $tmp = Join-Path $env:TEMP ("nes-bgm-" + [System.Guid]::NewGuid().ToString("N"))
    $ok = $false
    foreach ($url in @($GiteeRepo, $GitHubRepo)) {
        Say "clone" $url
        try {
            & git clone --depth 1 $url $tmp 2>$null
            if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        } catch {}
    }
    if (-not $ok) { Write-Host "下载源码失败，改用 -From <本地仓库>" -ForegroundColor Red; exit 1 }
    $src = Join-Path $tmp "skill\nes-bgm"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
Copy-Item -Recurse -Force $src $dest
Say "install" "skill -> $dest"

# ---- 4. 依赖 ---- #
Say "pip" "numpy / pretty_midi ..."
& $py -m pip install --upgrade pip 2>$null | Out-Null
& $py -m pip install -r (Join-Path $dest "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "依赖安装失败" -ForegroundColor Red; exit 1 }

& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Say "pip" "torch (CPU 版，约 200MB，稍等)"
    & $py -m pip install torch --index-url "https://download.pytorch.org/whl/cpu"
    if ($LASTEXITCODE -ne 0) { & $py -m pip install torch }
} else {
    Say "pip" "torch 已存在，跳过"
}

# ---- 5. 权重 ---- #
if ($NoModel) {
    Say "model" "跳过（已 -NoModel）"
} else {
    & $py (Join-Path $dest "scripts\download_model.py")
}

# ---- 6. 自检 ---- #
Say "check" "依赖自检"
& $py -c "import numpy, torch, pretty_midi; print('  numpy', numpy.__version__, '| torch', torch.__version__, '| pretty_midi ok')"
& $py (Join-Path $dest "scripts\bgm.py") --schema | Out-Null
if ($LASTEXITCODE -eq 0) { Say "check" "skill 可调用" }

Write-Host ""
Write-Host "安装完成  SKILL_DIR = $dest" -ForegroundColor Green
Write-Host "试一句：" -ForegroundColor Green
Write-Host "  & $py `"$dest\scripts\bgm.py`" --text `"地下洞窟，神秘，中速`" --bars 32 --out .\bgm_out"
