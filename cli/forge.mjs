#!/usr/bin/env node
/**
 * forge · 命令行入口
 * ------------------------------------------------------------------
 * 用法：
 *   forge verify                 运行全部数学不变量自检（38 项）
 *   forge bench [--json]         性能基准（GEMM / 量化 / 注意力 / 端到端）
 *   forge train [--steps N]      训练 tiny 模型并导出 GGUF
 *   forge chat <model.gguf>      与 GGUF 模型对话（生成）
 *   forge quantize <in> <out>    f32 GGUF → Q4_K / Q6_K / Q8_0 GGUF
 *   forge export <out.gguf>      随机初始化模型导出为 GGUF
 *   forge inspect <model.gguf>   打印 GGUF 元数据与张量清单
 *   forge test                   运行单元测试（node:test）
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const C = {
  b: (s) => `\x1b[1m${s}\x1b[0m`,
  d: (s) => `\x1b[2m${s}\x1b[0m`,
  g: (s) => `\x1b[32m${s}\x1b[0m`,
  y: (s) => `\x1b[33m${s}\x1b[0m`,
  r: (s) => `\x1b[31m${s}\x1b[0m`,
  c: (s) => `\x1b[36m${s}\x1b[0m`,
};

function parseArgs(argv) {
  const out = { _: [], flags: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) { out.flags[key] = next; i++; }
      else out.flags[key] = true;
    } else out._.push(a);
  }
  return out;
}

async function cmdVerify() {
  const { runVerify } = await import("../src/L10/verify.js");
  console.log(C.b("\nFORGE · 不变量自检 (Invariant Verification)\n"));
  const t0 = Date.now();
  const r = runVerify(false);
  for (const res of r.results) {
    const mark = res.pass ? C.g("✅") : C.r("❌");
    console.log(`${mark} ${C.c(res.id.padEnd(9))} ${res.name}  ${C.d("—")} ${C.d(res.detail)}`);
  }
  const dt = Date.now() - t0;
  console.log(`\n${r.fail === 0 ? C.g("全部通过") : C.r(`${r.fail} 项失败`)}: ${r.pass} 通过 / ${r.fail} 失败   ${C.d(`(${dt} ms)`)}\n`);
  process.exit(r.fail === 0 ? 0 : 1);
}

async function cmdBench() {
  const { runBench } = await import("../bench/bench.mjs");
  await runBench({ json: !!parseArgs(process.argv.slice(2)).flags.json });
}

async function cmdTrain(args) {
  const flags = parseArgs(process.argv.slice(2)).flags;
  const { trainTiny } = await import("../examples/03-train-and-generate.mjs");
  await trainTiny({
    steps: Number(flags.steps ?? 60),
    out: flags.out ?? join(ROOT, "out", "tiny-model.gguf"),
    vocabText: flags.text,
  });
}

async function cmdChat(args) {
  const flags = parseArgs(process.argv.slice(2)).flags;
  const path = args._[0];
  if (!path || !existsSync(path)) {
    console.error(C.r("用法: forge chat <model.gguf> --prompt \"...\" [--tokens N] [--temp T]"));
    process.exit(1);
  }
  const { loadGGUF, generate } = await import("../src/L9/engine.js");
  const bytes = readFileSync(path);
  const model = loadGGUF(new Uint8Array(bytes));
  console.log(C.d(`模型: ${model.metadata.get("general.name") ?? path}  vocab=${model.config.vocabSize} layers=${model.config.nLayers} dim=${model.config.dim}`));
  const prompt = flags.prompt ?? args._[1] ?? "";
  const maxTokens = Number(flags.tokens ?? 128);
  const t0 = Date.now();
  const out = generate(model, {
    prompt,
    maxTokens,
    temperature: flags.temp !== undefined ? Number(flags.temp) : 0.8,
    top_p: flags.topp !== undefined ? Number(flags.topp) : 0.95,
    min_p: flags.minp !== undefined ? Number(flags.minp) : 0.05,
    top_k: flags.topk !== undefined ? Number(flags.topk) : 40,
    seed: Number(flags.seed ?? 0),
  });
  const dt = Date.now() - t0;
  console.log(C.b("\n── 生成结果 ──\n"));
  console.log(out);
  console.log(C.d(`\n(${maxTokens} tokens 上限, ${dt} ms)`));
}

async function cmdQuantize(args) {
  const flags = parseArgs(process.argv.slice(2)).flags;
  const inPath = args._[0], outPath = args._[1];
  if (!inPath || !outPath) {
    console.error(C.r("用法: forge quantize <in.gguf> <out.gguf> [--type 12|14|8|1]"));
    process.exit(1);
  }
  const { quantizeGGUF } = await import("../tools/quantize-gguf.mjs");
  const r = await quantizeGGUF(inPath, outPath, { type: Number(flags.type ?? 12) });
  console.log(C.g(`✓ ${r.inPath} → ${r.outPath}`));
  console.log(`  类型 ${r.type} (${r.bpw} bpw) · 量化张量 ${r.quantizedTensors} 个`);
  console.log(`  权重 ${(r.origBytes / 1048576).toFixed(2)} MB → ${(r.newBytes / 1048576).toFixed(2)} MB (${r.ratio.toFixed(2)}×)`);
}

async function cmdExport(args) {
  const flags = parseArgs(process.argv.slice(2)).flags;
  const outPath = args._[0] ?? join(ROOT, "out", "random-model.gguf");
  const { exportGGUF } = await import("../src/L9/engine.js");
  const { Config } = await import("../src/L6/transformer.js");
  const { initWeights } = await import("../src/L6/train.js");
  const cfg = new Config({
    dim: Number(flags.dim ?? 256), hidden: Number(flags.hidden ?? 512),
    nLayers: Number(flags.layers ?? 4), nHeads: Number(flags.heads ?? 8),
    nKvHeads: Number(flags.kvheads ?? 4), vocabSize: Number(flags.vocab ?? 4096),
    maxSeq: Number(flags.ctx ?? 1024),
  });
  const w = initWeights(cfg, Number(flags.seed ?? 42));
  const bytes = exportGGUF(w, cfg, null, 0, flags.name ?? "forge-random");
  writeFileSync(outPath, bytes);
  console.log(C.g(`✓ 已导出 ${outPath}  (${(bytes.length / 1048576).toFixed(2)} MB, ${cfg.nLayers} 层, dim=${cfg.dim})`));
}

async function cmdInspect(args) {
  const path = args._[0];
  if (!path || !existsSync(path)) { console.error(C.r("用法: forge inspect <model.gguf>")); process.exit(1); }
  const { readGGUF } = await import("../src/L9/gguf.js");
  const { TYPE_INFO } = await import("../src/L3/quant.js");
  const g = readGGUF(new Uint8Array(readFileSync(path)));
  console.log(C.b("\n── GGUF 元数据 ──\n"));
  for (const [k, v] of g.meta) {
    const s = Array.isArray(v) ? `[${v.length} 项] ${JSON.stringify(v.slice(0, 8))}${v.length > 8 ? " …" : ""}` : JSON.stringify(v);
    console.log(`  ${C.c(k.padEnd(46))} ${s}`);
  }
  console.log(C.b(`\n── 张量 (${g.tensors.length}) ──\n`));
  for (const t of g.tensors) {
    const ti = TYPE_INFO[t.type];
    const nelem = t.dims.reduce((a, b) => a * b, 1);
    const bytes = ti ? Math.ceil(nelem / ti.blockSize) * ti.blockBytes : 0;
    console.log(`  ${t.name.padEnd(38)} ${String(t.dims.join("×")).padEnd(18)} ${(ti?.name ?? `type${t.type}`).padEnd(6)} ${(bytes / 1024).toFixed(1)} KiB`);
  }
  console.log();
}

async function cmdTest() {
  const { spawnSync } = await import("node:child_process");
  const r = spawnSync(process.execPath, ["--test", "test/"], { cwd: ROOT, stdio: "inherit" });
  process.exit(r.status ?? 1);
}

const HELP = `
${C.b("FORGE")} · Verifiable AI Kernel  ${C.d("v1.0.0 · 晨星")}

${C.b("命令")}：
  ${C.c("verify")}                运行全部数学不变量自检
  ${C.c("bench")} [--json]        性能基准（GEMM/量化/注意力/端到端）
  ${C.c("train")} [--steps N] [--out F]   训练 tiny 模型并导出 GGUF
  ${C.c("chat")} <model.gguf> --prompt "..." [--tokens N] [--temp T] [--seed N]
  ${C.c("quantize")} <in.gguf> <out.gguf> [--type 12|14|8|1]   f32 → 量化 GGUF
  ${C.c("export")} <out.gguf> [--dim N] [--layers N]         导出随机初始化 GGUF
  ${C.c("inspect")} <model.gguf>  打印 GGUF 元数据与张量清单
  ${C.c("test")}                  运行 node:test 单元测试

${C.b("示例")}：
  node cli/forge.mjs verify
  node cli/forge.mjs bench --json
  node cli/forge.mjs train --steps 60 --out out/tiny.gguf
  node cli/forge.mjs chat out/tiny.gguf --prompt "hello" --tokens 64
`;

const cmd = process.argv[2];
const args = parseArgs(process.argv.slice(3)); // { _: 位置参数, flags: --选项 }
try {
  switch (cmd) {
    case "verify": await cmdVerify(); break;
    case "bench": await cmdBench(); break;
    case "train": await cmdTrain(args); break;
    case "chat": await cmdChat(args); break;
    case "quantize": await cmdQuantize(args); break;
    case "export": await cmdExport(args); break;
    case "inspect": await cmdInspect(args); break;
    case "test": await cmdTest(); break;
    case undefined:
    case "help":
    case "-h":
    case "--help": console.log(HELP); break;
    default:
      console.error(C.r(`未知命令: ${cmd}`));
      console.log(HELP);
      process.exit(1);
  }
} catch (e) {
  console.error(C.r(`✗ ${e.message}`));
  if (process.env.FORGE_DEBUG) console.error(e.stack);
  process.exit(1);
}
