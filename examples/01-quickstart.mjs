/**
 * 示例 01 · 五分钟上手：加载 GGUF 模型并生成文本
 * ------------------------------------------------------------------
 * 演示 forge-llm 的最小可用路径：
 *   1) 用 CLI 训练一个 tiny 字符级模型（30 秒）
 *   2) 加载 GGUF → 推理 → 采样生成
 *   3) 对比 greedy / temperature 两种采样
 *
 * 运行：node examples/01-quickstart.mjs
 * （若 out/tiny-model.gguf 不存在，会自动先训练一个）
 */

import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = path.join(ROOT, "out", "tiny-model.gguf");

if (!existsSync(MODEL)) {
  console.log("未找到模型，先训练一个 tiny 模型（约 30 秒）…\n");
  execFileSync(process.execPath, [path.join(ROOT, "cli", "forge.mjs"), "train", "--steps", "60", "--out", MODEL], { stdio: "inherit" });
}

const { loadGGUF, generate } = await import("../src/L9/engine.js");
const { readFileSync } = await import("node:fs");

console.log("\n\x1b[1m── 加载模型 ──\x1b[0m");
const model = loadGGUF(new Uint8Array(readFileSync(MODEL)));
const { config } = model;
console.log(`  dim=${config.dim} layers=${config.nLayers} heads=${config.nHeads}/${config.nKvHeads} vocab=${config.vocabSize}`);

const PROMPT = "the ";
console.log(`\n\x1b[1m── 生成（prompt = ${JSON.stringify(PROMPT)}）──\x1b[0m`);

// greedy：确定性，可复现
const greedy = generate(model, { prompt: PROMPT, maxTokens: 60, temperature: 0, top_k: 1, seed: 0 });
console.log(`\n  \x1b[1mgreedy (temperature=0)\x1b[0m\n  ${JSON.stringify(greedy)}`);

// 采样：temperature + top-k + min-p
const sampled = generate(model, { prompt: PROMPT, maxTokens: 60, temperature: 0.8, top_k: 40, min_p: 0.05, seed: 42 });
console.log(`\n  \x1b[1msampled (t=0.8, top_k=40, min_p=0.05)\x1b[0m\n  ${JSON.stringify(sampled)}`);

// 可复现性：同种子必须同输出
const again = generate(model, { prompt: PROMPT, maxTokens: 60, temperature: 0.8, top_k: 40, min_p: 0.05, seed: 42 });
console.log(`\n  \x1b[1m可复现性\x1b[0m 同种子重放一致：${sampled === again ? "✅" : "❌"}`);

console.log("\n提示：模型只在 574 字符的演示语料上训练过，输出是「学过的统计规律」而非语义理解。");
console.log("      完整训练-导出-量化闭环见 examples/03-train-and-generate.mjs。\n");
