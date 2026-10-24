#!/usr/bin/env bash
# forge-llm · 一键安装与验证
#
# 本项目零依赖零构建 —— 「安装」= 检查 Node 版本 + 跑一遍不变量自检。
# 任何一条不变量失败即返回非 0 退出码，说明环境或内核有问题。

set -euo pipefail

RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

cd "$(dirname "$0")"

echo -e "${BOLD}FORGE · Verifiable AI Kernel${RESET}"
echo -e "${DIM}零依赖纯 JavaScript LLM 内核 · 作者 晨星${RESET}"
echo ""

# ── 1. 环境检查 ──
echo -e "${CYAN}[1/4]${RESET} 检查 Node 版本…"
if ! command -v node >/dev/null 2>&1; then
  echo -e "${RED}✗ 未找到 node。请安装 Node.js ≥ 20：https://nodejs.org${RESET}"
  exit 1
fi
NODE_VER=$(node -v)
NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo -e "${RED}✗ Node $NODE_VER 版本过低（需要 ≥ 20）${RESET}"
  exit 1
fi
echo -e "  ${GREEN}✓${RESET} Node $NODE_VER"

# ── 2. 依赖 ──
echo -e "${CYAN}[2/4]${RESET} 检查依赖…"
if [ -f package.json ]; then
  DEPS=$(node -p "Object.keys((require('./package.json').dependencies||{})).length" 2>/dev/null || echo 0)
  if [ "$DEPS" -eq 0 ]; then
    echo -e "  ${GREEN}✓${RESET} 零运行时依赖 —— 无需 npm install"
  else
    echo -e "  ${YELLOW}!${RESET} 检测到 $DEPS 个依赖，正在安装…"
    npm install --no-audit --no-fund
  fi
fi

# ── 3. 不变量自检 ──
echo -e "${CYAN}[3/4]${RESET} 运行数学不变量自检…"
if node cli/forge.mjs verify; then
  echo -e "  ${GREEN}✓${RESET} 全部不变量通过"
else
  echo -e "  ${RED}✗ 不变量验证失败 —— 内核不可信，请勿继续${RESET}"
  exit 1
fi

# ── 4. 单元测试 ──
echo -e "${CYAN}[4/4]${RESET} 运行单元测试…"
if node --test "test/*.test.js" 2>&1 | tail -6; then
  echo -e "  ${GREEN}✓${RESET} 测试通过"
else
  echo -e "  ${RED}✗ 测试失败${RESET}"
  exit 1
fi

cat <<'EOF'

────────────────────────────────────────────
安装完成。试试：

  node cli/forge.mjs bench
  node cli/forge.mjs train --steps 200 --out out/tiny.gguf
  node cli/forge.mjs chat out/tiny.gguf --prompt "the "

────────────────────────────────────────────
EOF
