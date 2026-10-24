/**
 * 示例 02 · 量化：f32 GGUF → Q4_K/Q6_K/Q8_0，精度与体积的量化交易
 * ------------------------------------------------------------------
 * 演示 forge-llm 的量化闭环（位布局对齐 llama.cpp）：
 *   1) 导出随机初始化 f32 模型
 *   2) 分别量化为 Q8_0 / Q6_K / Q4_K
 *   3) 对比体积、压缩比、推理输出一致性（cos / argmax）
 *
 * 运行：node examples/02-quantize.mjs
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "out");
mkdirSync(OUT, { recursive: true });

const { exportGGUF, loadGGUF, generate } = await import("../src/L9/engine.js");
const { Config } = await import("../src/L6/transformer.js");
const { initWeights } = await import("../src/L6/train.js");
const { KVCache } = await import("../src/L5/attention.js");
const { quantizeGGUF } = await import("../tools/quantize-gguf.mjs");

// 维度取真实模型量级：行长为 512/1024，是 k-quant 块大小(256)的整数倍
const cfg = new Config({ dim: 512, hidden: 1024, nLayers: 2, nHeads: 8, nKvHeads: 4, vocabSize: 512, maxSeq: 64 });
console.log(`模型配置：dim=${cfg.dim} layers=${cfg.nLayers} heads=${cfg.nHeads}/${cfg.nKvHeads} vocab=${cfg.vocabSize}`);

// 1) f32 基线
const W = initWeights(cfg, 20260101);
const f32Path = path.join(OUT, "ex2-f32.gguf");
writeFileSync(f32Path, exportGGUF(W, cfg, null, 0, "ex2"));
const f32 = loadGGUF(new Uint8Array(readFileSync(f32Path)));
const baseBytes = readFileSync(f32Path).byteLength;
console.log(`f32 基线：${(baseBytes / 1024).toFixed(1)} KB\n`);

// 2) 三种量化
const rows = [];
for (const [name, type] of [["Q8_0", 8], ["Q6_K", 14], ["Q4_K", 12]]) {
  const p = path.join(OUT, `ex2-${name}.gguf`);
  await quantizeGGUF(f32Path, p, { type });
  const m = loadGGUF(new Uint8Array(readFileSync(p)));
  const bytes = readFileSync(p).byteLength;
  rows.push({ name, bytes, ratio: baseBytes / bytes, model: m, cos: 0 });
}

// 3) 一致性：同一段 prompt，比较 logits 余弦与 argmax
console.log("量化格式   体积      压缩比    logits cos   argmax 一致");
const PROMPT = [1, 5, 9, 13, 17, 21, 25, 29];
const cacheF = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
let refLogits;
const ref = [];
for (let i = 0; i < PROMPT.length; i++) refLogits = f32.transformer.forward(PROMPT[i], i, cacheF);
for (let i = 0; i < 8; i++) refLogits = f32.transformer.forward(i % cfg.vocabSize, PROMPT.length + i, cacheF);

let allMatch = true;
for (const r of rows) {
  const cache = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
  let lg;
  for (let i = 0; i < PROMPT.length; i++) lg = r.model.transformer.forward(PROMPT[i], i, cache);
  for (let i = 0; i < 8; i++) lg = r.model.transformer.forward(i % cfg.vocabSize, PROMPT.length + i, cache);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < cfg.vocabSize; i++) { dot += refLogits[i] * lg[i]; na += refLogits[i] ** 2; nb += lg[i] ** 2; }
  const cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
  const am = argmax(refLogits, cfg.vocabSize) === argmax(lg, cfg.vocabSize);
  if (!am) allMatch = false;
  r.cos = cos.toFixed(5);
  console.log(`${r.name.padEnd(9)} ${(r.bytes / 1024).toFixed(1).padStart(6)} KB ${r.ratio.toFixed(2).padStart(6)}×   ${r.cos.padStart(8)}   ${am ? "✅" : "❌"}`);
}
const q4 = rows.find((r) => r.name === "Q4_K");
console.log(`\n结论：Q4_K 以 ${q4.ratio.toFixed(2)}× 压缩换取 cos=${q4.cos} 的精度损失——量化是"可测量的误差换可测量的体积"。`);
console.log("注：token_embd 与输出头按 llama.cpp 惯例保持 f32，故文件级压缩比低于权重理论值（Q4_K 4.5 bpw ≈ 7.1×）。");
console.log(allMatch ? "三种量化的 argmax 均与 f32 一致。\n" : "存在 argmax 翻转（近平手时属正常现象）。\n");

function argmax(a, n) { let bi = 0; for (let i = 1; i < n; i++) if (a[i] > a[bi]) bi = i; return bi; }
