/**
 * 示例 03 · 训练 tiny 模型 → 导出 GGUF → 加载 → 生成
 * ------------------------------------------------------------------
 * 一条命令跑完整条链路，验证「训练 → 量化 → 推理 → GGUF 互操作」全通：
 *
 *   node examples/03-train-and-generate.mjs
 *   node cli/forge.mjs train --steps 80 --out out/tiny.gguf
 *
 * 训练语料：默认用内置的小语料（可 --text 传入自定义文件）。
 * 模型规模刻意取小（dim=64, 2 层），让 CPU 上几十秒内能收敛到可读的输出。
 */

import { writeFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Config } from "../src/L6/transformer.js";
import { TrainableTransformer, initWeights, train } from "../src/L6/train.js";
import { CharTokenizer } from "../src/L7/tokenizer.js";
import { exportGGUF, loadGGUF, generate } from "../src/L9/engine.js";
import { resetTape, Adam } from "../src/L2/autograd.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** 内置训练语料：英文短句，字符级分词足以学到结构 */
const DEFAULT_TEXT = `
the forge kernel verifies every mathematical invariant before it runs.
a kernel that cannot prove its own correctness is not a kernel but a guess.
numerical precision is a feature not an accident.
quantization trades a measurable amount of error for a measurable amount of speed.
attention scales quadratically with sequence length unless you are clever.
the best way to verify a system is to make the system verify itself.
deterministic randomness is the foundation of reproducible benchmarks.
every layer of the stack exposes its own contract as an executable assertion.
`;

function buildTokenizer(text) {
  // CharTokenizer 需要 bos/eos 特殊 token；这里 vocab 从 3 开始
  return new CharTokenizer(text, { bos: 1, eos: 2 });
}

export async function trainTiny({ steps = 60, out = "out/tiny-model.gguf", vocabText, quiet = false } = {}) {
  const text = vocabText && existsSync(vocabText) ? readFileSync(vocabText, "utf-8") : DEFAULT_TEXT;
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  if (!quiet) {
    console.log(`\n${"\x1b[1m"}FORGE · tiny 模型训练${"\x1b[0m"}`);
    console.log(`  语料 ${text.length} 字符 → ${data.length} token，词表 ${tok.vocabSize}`);
  }

  const cfg = new Config({
    dim: 64, hidden: 128, nLayers: 2, nHeads: 4, nKvHeads: 2,
    vocabSize: tok.vocabSize, maxSeq: 64, normEps: 1e-5,
  });

  const model = new TrainableTransformer(cfg, initWeights(cfg, 20260101));
  if (!quiet) {
    console.log(`  模型 dim=${cfg.dim} hidden=${cfg.hidden} layers=${cfg.nLayers} heads=${cfg.nHeads}/${cfg.nKvHeads} vocab=${cfg.vocabSize}`);
    console.log(`  参数量 ${model.countParams().toLocaleString()}`);
  }

  // ── 记忆化检验：训练闭环正确性的最强证据 ──
  // 一个正确实现的训练循环必须能把 loss 压到接近 0；此步失败即说明 AD/优化器有 bug。
  if (!quiet) {
    console.log("\n\x1b[2m── 记忆化检验（训练闭环正确性）──\x1b[0m");
    const memo = new TrainableTransformer(cfg, initWeights(cfg, 20260101));
    const mdata = tok.encode("the forge kernel verifies every invariant.\n");
    const opt = new Adam(memo.paramList, { lr: 0.02, totalSteps: 150, clip: 1.0 });
    const l0 = (() => { resetTape(); const l = memo.loss(mdata, mdata.slice(1)); return memo.backward(l); })();
    for (let i = 0; i < 150; i++) {
      resetTape(); memo.zeroGrad();
      const l = memo.loss(mdata, mdata.slice(1));
      memo.backward(l);
      opt.step();
    }
    resetTape();
    const l1 = (() => { const l = memo.loss(mdata, mdata.slice(1)); return memo.backward(l); })();
    const ok = l1 < 0.05;
    console.log(`  loss ${l0.toFixed(4)} → ${l1.toFixed(6)}   ${ok ? "\x1b[32m✓ 收敛到近 0（训练闭环正确）\x1b[0m" : "\x1b[31m✗ 未收敛\x1b[0m"}`);
    if (!ok) throw new Error("记忆化检验失败 —— 训练闭环存在 bug");
  }

  // ── 训练 ──
  if (!quiet) console.log("\n\x1b[2m── 训练 ──\x1b[0m");
  const history = train(model, data, {
    steps, batchSize: 8, seqLen: 32, lr: 5e-3, logEvery: Math.max(1, Math.floor(steps / 6)),
  });

  // ── 导出 GGUF ──
  const weights = model.exportWeights();
  // 词表需转成 GGUFVocab（字节串形式）
  const tokenBytes = [];
  for (let i = 0; i < tok.vocabSize; i++) {
    tokenBytes.push(new TextEncoder().encode(tok.itos[i] ?? ""));
  }
  const { GGUFVocab } = await import("../src/L7/tokenizer.js");
  const vocab = new GGUFVocab(tokenBytes, new Array(tok.vocabSize).fill(0), new Array(tok.vocabSize).fill(0),
    { bos: 1, eos: 2, addBos: false, addEos: false });

  const bytes = exportGGUF(weights, cfg, vocab, 0, "forge-tiny");
  const dir = dirname(out);
  if (dir && !existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(out, bytes);
  if (!quiet) {
    console.log(`\n\x1b[2m── 导出 ──\x1b[0m`);
    console.log(`  GGUF → ${out}  (${(bytes.length / 1024).toFixed(1)} KiB)`);
  }

  // ── 重新加载并生成（验证训练权重真的落盘且可推理）──
  const model2 = loadGGUF(new Uint8Array(readFileSync(out)));
  if (!quiet) {
    console.log(`\n\x1b[2m── 重新加载并生成 ──\x1b[0m`);
    console.log(`  加载成功：dim=${model2.config.dim} layers=${model2.config.nLayers} vocab=${model2.config.vocabSize}`);
    const prompt = "the ";
    const t0 = Date.now();
    const gen = generate(model2, { prompt, maxTokens: 120, temperature: 0.7, top_p: 0.95, min_p: 0.05, seed: 42 });
    const dt = Date.now() - t0;
    console.log(`\n  prompt: ${JSON.stringify(prompt)}`);
    console.log(`  ${"\x1b[36m"}${gen.replace(/\n/g, " ")}${"\x1b[0m"}`);
    console.log(`\x1b[2m  (${dt} ms)\x1b[0m`);
  }

  return { cfg, model, history, out, bytes };
}

if (process.argv[1] && process.argv[1].replace(/\\/g, "/").endsWith("examples/03-train-and-generate.mjs")) {
  const idx = process.argv.indexOf("--steps");
  const steps = idx > 0 ? Number(process.argv[idx + 1]) : 60;
  const oidx = process.argv.indexOf("--out");
  const out = oidx > 0 ? process.argv[oidx + 1] : join(__dirname, "..", "out", "tiny-model.gguf");
  await trainTiny({ steps, out });
}
