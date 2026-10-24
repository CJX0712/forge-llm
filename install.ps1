# forge-llm · 一键安装与验证（Windows PowerShell）
#
# 本项目零依赖零构建 —— 「安装」= 检查 Node 版本 + 跑一遍不变量自检。

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host $msg -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  $msg" -ForegroundColor Red }

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host ""
Write-Host "FORGE · Verifiable AI Kernel" -ForegroundColor White
Write-Host "零依赖纯 JavaScript LLM 内核 · 作者 晨星" -ForegroundColor DarkGray
Write-Host ""

# ── 1. 环境检查 ──
Step "[1/4] 检查 Node 版本…"
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
  Fail "未找到 node。请安装 Node.js ≥ 20：https://nodejs.org"
  exit 1
}
$nodeVer = node -v
$nodeMajor = [int](node -p "process.versions.node.split('.')[0]")
if ($nodeMajor -lt 20) {
  Fail "Node $nodeVer 版本过低（需要 ≥ 20）"
  exit 1
}
Ok "Node $nodeVer"

# ── 2. 依赖 ──
Step "[2/4] 检查依赖…"
if (Test-Path package.json) {
  $deps = node -p "Object.keys((require('./package.json').dependencies||{})).length"
  if ([int]$deps -eq 0) {
    Ok "零运行时依赖 —— 无需 npm install"
  } else {
    Write-Host "  检测到 $deps 个依赖，正在安装…"
    npm install --no-audit --no-fund
  }
}

# ── 3. 不变量自检 ──
Step "[3/4] 运行数学不变量自检…"
node cli/forge.mjs verify
if ($LASTEXITCODE -ne 0) {
  Fail "不变量验证失败 —— 内核不可信，请勿继续"
  exit 1
}
Ok "全部不变量通过"

# ── 4. 单元测试 ──
Step "[4/4] 运行单元测试…"
node --test "test/*.test.js"
if ($LASTEXITCODE -ne 0) {
  Fail "测试失败"
  exit 1
}
Ok "测试通过"

Write-Host ""
Write-Host "────────────────────────────" -ForegroundColor DarkGray
Write-Host "安装完成。试试：" -ForegroundColor White
Write-Host "  node cli/forge.mjs bench"
Write-Host "  node cli/forge.mjs train --steps 200 --out out/tiny.gguf"
Write-Host "  node cli/forge.mjs chat out/tiny.gguf --prompt `"the `"" -ForegroundColor DarkGray
Write-Host "────────────────────────────" -ForegroundColor DarkGray
