/**
 * run_interop.mjs —— 一条命令跑完「GGUF 外部证伪」全流程
 *
 *   1. 用 llama.cpp 官方 gguf-py 生成官方命名的 GGUF（权重不经 forge 之手）
 *   2. numpy_forward.py  —— 独立 numpy 实现同一套前向数学（对照组）
 *   3. dump_stages.mjs   —— forge 逐阶段中间量（用于定位误差环节）
 *   4. verify_engine.mjs —— forge「出厂代码路径」前向（用于证明发布代码正确）
 *   5. cmp_full.py       —— 逐阶段对拍（embed/afterAttn/afterFFN/xFinal/logits）
 *   6. cmp_logits.py     —— 出厂前向 vs numpy 参考的 logits 对拍
 *   7. verify_export.*    —— 反向闭环：forge 写出的 GGUF 由官方 GGUFReader 读回
 *   8. verify_quant.*    —— 量化位布局：forge 量化字节由官方 dequantize 读回
 *   7. 反向闭环          —— forge 写出的 GGUF 由 llama.cpp 官方 gguf-py 读回并逐张量比对
 *                           （prove「写」侧也符合官方格式，而非只「读」侧兼容）
 *
 * 用法（在仓库根目录）：
 *   node tools/interop/run_interop.mjs            # 正常流程
 *   node tools/interop/run_interop.mjs --force    # 强制重新生成 GGUF
 *   FORGE_PYTHON=/path/to/python node tools/interop/run_interop.mjs   # 指定解释器
 *
 * 退出码：0 = 全部 PASS，1 = 有环节失败
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const OUT = join(HERE, "out");
const GGUF = join(OUT, "llama-official.gguf");
const NPZ = GGUF + ".npz";
const FORCE = process.argv.includes("--force");

const log = (s) => process.stdout.write(s + "\n");
const hr = (t) => log(`\n\x1b[36m── ${t} ${"─".repeat(Math.max(0, 58 - t.length))}\x1b[0m`);

/** 探测可用的 Python 解释器：FORGE_PYTHON > python > py -3 */
function findPython() {
  const cands = [];
  if (process.env.FORGE_PYTHON) cands.push({ cmd: process.env.FORGE_PYTHON, args: [] });
  cands.push({ cmd: "python", args: [] }, { cmd: "py", args: ["-3"] });
  for (const c of cands) {
    // 必须 shell:false —— Windows 的 cmd.exe 会把 -c "import numpy;..." 里的分号当命令分隔符
    const r = spawnSync(c.cmd, [...c.args, "-c", "import numpy;print(numpy.__version__)"], {
      encoding: "utf-8",
    });
    if (r.status === 0) return { ...c, numpy: (r.stdout || "").trim() };
  }
  return null;
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, {
    cwd: ROOT, encoding: "utf-8", stdio: "inherit", ...opts,
  });
  return r.status === 0;
}

log("\x1b[1mforge-llm · GGUF 外部证伪（interop falsification）\x1b[0m");
log("证据链：官方 gguf-py 产权重 → 独立 numpy 对照 → forge 出厂前向 → 逐阶段对拍");
mkdirSync(OUT, { recursive: true });

// ── 0) 依赖检查 ────────────────────────────────────────────────────────
const py = findPython();
if (!py) {
  log("\x1b[31m未找到带 numpy 的 Python。请安装 Python3 + numpy，或设置 FORGE_PYTHON。\x1b[0m");
  process.exit(1);
}
log(`Python: ${py.cmd} ${py.args.join(" ")}  (numpy ${py.numpy})`);

const GGUFPY = process.env.FORGE_GGUFPY || join(ROOT, ".ref", "gguf-py");
if (!existsSync(join(GGUFPY, "gguf", "gguf_writer.py"))) {
  log(`\x1b[31m缺少 llama.cpp 官方 gguf-py。获取方式：\x1b[0m`);
  log("  git clone --depth 1 https://github.com/ggml-org/llama.cpp .ref/llama.cpp");
  log("  cp -r .ref/llama.cpp/gguf-py .ref/gguf-py        # 或设置 FORGE_GGUFPY=/path/to/gguf-py");
  process.exit(1);
}
log(`gguf-py: ${GGUFPY}`);

// ── 1) 官方 GGUF ───────────────────────────────────────────────────────
hr("1/8 生成官方 GGUF（llama.cpp 官方工具链）");
if (!FORCE && existsSync(GGUF) && existsSync(NPZ)) {
  log(`复用已有 ${GGUF}（${statSync(GGUF).size} 字节）；加 --force 可重建`);
} else {
  if (!run(py.cmd, [...py.args, "tools/interop/make_official_gguf.py", "tools/interop/out/llama-official.gguf"])) {
    log("\x1b[31mGGUF 生成失败\x1b[0m"); process.exit(1);
  }
}

// ── 2) numpy 参考 ──────────────────────────────────────────────────────
hr("2/8 独立 numpy 参考前向");
if (!run(py.cmd, [...py.args, "tools/interop/numpy_forward.py", "tools/interop/out/llama-official.gguf.npz",
  "--ids", "2,3,4,5", "--out", "tools/interop/out/np-logits.json"])) {
  log("\x1b[31mnumpy 参考前向失败\x1b[0m"); process.exit(1);
}

// ── 3) forge 逐阶段 dump ───────────────────────────────────────────────
hr("3/8 forge 逐阶段中间量");
if (!run(process.execPath, ["tools/interop/dump_stages.mjs"])) {
  log("\x1b[31m逐阶段 dump 失败\x1b[0m"); process.exit(1);
}

// ── 4) forge 出厂前向 ──────────────────────────────────────────────────
hr("4/8 forge 出厂代码路径前向（Transformer.forward + KVCache）");
if (!run(process.execPath, ["tools/interop/verify_engine.mjs"])) {
  log("\x1b[31m出厂前向失败\x1b[0m"); process.exit(1);
}

// ── 5) 逐阶段对拍 ──────────────────────────────────────────────────────
hr("5/8 逐阶段对拍");
const okFull = run(py.cmd, [...py.args, "tools/interop/cmp_full.py"]);

// ── 6) logits 对拍 ─────────────────────────────────────────────────────
hr("6/8 logits 对拍");
const okLogits = run(py.cmd, [...py.args, "tools/interop/cmp_logits.py"]);

// ── 7) 反向闭环：forge 写 → 官方 gguf-py 读 ────────────────────────────
hr("7/8 反向互操作（forge 写出的 GGUF 由官方 gguf-py 读回）");
let okExport = false;
if (run(process.execPath, ["cli/forge.mjs", "export", "tools/interop/out/forge-export.gguf",
  "--dim", "64", "--layers", "2", "--vocab", "256"])) {
  if (run(process.execPath, ["tools/interop/verify_export.mjs"])) {
    okExport = run(py.cmd, [...py.args, "tools/interop/verify_export.py"]);
  }
}

// ── 8) 量化位布局：forge 量化 → 官方 gguf-py 反量化 ────────────────────
hr("8/8 量化位布局对拍（官方 dequantize 读 forge 字节）");
let okQuant = false;
if (run(process.execPath, ["tools/interop/verify_quant.mjs"])) {
  okQuant = run(py.cmd, [...py.args, "tools/interop/verify_quant.py"]);
}

hr("结论");
const all = okFull && okLogits && okExport && okQuant;
if (all) {
  log("\x1b[32m\x1b[1m全部 PASS ✅\x1b[0m  forge-llm 的 GGUF 读写、前向数学与量化位布局与 llama.cpp 生态一致");
  process.exit(0);
}
log("\x1b[31m\x1b[1m存在失败项 ❌\x1b[0m  逐阶段: " + (okFull ? "PASS" : "FAIL")
  + " / logits: " + (okLogits ? "PASS" : "FAIL")
  + " / 反向互操作: " + (okExport ? "PASS" : "FAIL")
  + " / 量化布局: " + (okQuant ? "PASS" : "FAIL"));
process.exit(1);
