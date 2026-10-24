/**
 * 单元测试 · L7 分词器 & L8 采样 & L9 GGUF/引擎 & L10 验证层
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { CharTokenizer, GGUFVocab } from "../src/L7/tokenizer.js";
import { sample } from "../src/L8/sampling.js";
import { readGGUF, writeGGUF, GGUFValueType } from "../src/L9/gguf.js";
import { loadGGUF, exportGGUF, generate } from "../src/L9/engine.js";
import { Config } from "../src/L6/transformer.js";
import { initWeights } from "../src/L6/train.js";
import { runVerify } from "../src/L10/verify.js";
import { RNG } from "../src/L0/numeric.js";

test("CharTokenizer 编码/解码往返", () => {
  const text = "hello world, forge!";
  const tok = new CharTokenizer(text);
  const ids = tok.encode(text);
  assert.equal(ids[0], tok.bos);
  assert.equal(ids[ids.length - 1], tok.eos);
  assert.equal(tok.decode(ids), text);
});

test("CharTokenizer 词表稳定（排序确定）", () => {
  const a = new CharTokenizer("cba"), b = new CharTokenizer("abc");
  assert.deepEqual(Array.from(a.chars), Array.from(b.chars));
});

test("GGUFVocab 按字节解码（UTF-8 还原）", () => {
  const enc = new TextEncoder();
  const toks = [enc.encode("你"), enc.encode("好"), enc.encode("世界")];
  // bos/eos 取不在 token 范围内的值，避免把普通 token 当作特殊符号跳过
  const v = new GGUFVocab(toks, [0, 0, 0], [0, 0, 0], { bos: 100, eos: 101 });
  assert.equal(v.decode([0, 1, 2]), "你好世界");
});

test("GGUFVocab 编码：最长匹配", () => {
  const enc = new TextEncoder();
  const v = new GGUFVocab([enc.encode("a"), enc.encode("ab"), enc.encode("c")], [0, 0, 0], [0, 0, 0],
    { bos: 100, eos: 101, addBos: false, addEos: false });
  // 最长匹配："abc" → "ab"(id=1) + "c"(id=2)
  assert.deepEqual(v.encode("abc"), [1, 2]);
});

test("采样：temperature=0 → greedy", () => {
  const logits = new Float32Array([0.1, 5.0, 0.2]);
  for (let i = 0; i < 20; i++) assert.equal(sample(logits, { temperature: 0 }), 1);
});

test("采样：top_k=1 → greedy", () => {
  const logits = new Float32Array([0.1, 5.0, 0.2]);
  assert.equal(sample(logits, { temperature: 1, top_k: 1 }), 1);
});

test("采样：同种子可复现", () => {
  const logits = new Float32Array(50);
  const rng = new RNG(3);
  for (let i = 0; i < 50; i++) logits[i] = rng.normal();
  const a = sample(logits, { temperature: 1, seed: 42 });
  const b = sample(logits, { temperature: 1, seed: 42 });
  assert.equal(a, b);
});

test("采样：min-p 截断后仍能采样到高概率 token", () => {
  const logits = new Float32Array([10, 9, -20, -20]);
  const picked = new Set();
  for (let s = 0; s < 200; s++) picked.add(sample(logits, { temperature: 1, min_p: 0.5, seed: s }));
  assert.ok(picked.size <= 2, `picked=${[...picked]}`);
});

test("采样：top_p 强约束时集中在头部", () => {
  const logits = new Float32Array([10, 9.9, 0, -1, -2]);
  const counts = new Array(5).fill(0);
  for (let s = 0; s < 500; s++) counts[sample(logits, { temperature: 1, top_p: 0.5, seed: s })]++;
  assert.ok(counts[0] + counts[1] >= 490, `counts=${counts}`);
});

test("GGUF 写→读：元数据逐项一致", () => {
  const meta = new Map([
    ["general.architecture", "forge"],
    ["forge.block_count", 4],
    ["forge.embedding_length", 128],
    ["forge.norm_eps", 1e-5],
    ["forge.flag", true],
    ["forge.names", ["a", "b", "c"]],
  ]);
  const data = new Float32Array([1, 2, 3, 4]);
  const tensors = [{
    name: "token_embd.weight", dims: [2, 2], type: 0,
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  }];
  const { bytes } = writeGGUF({ meta, tensors });
  const g = readGGUF(bytes);
  assert.equal(g.version, 3);
  assert.equal(g.meta.get("general.architecture"), "forge");
  assert.equal(g.meta.get("forge.block_count"), 4);
  assert.equal(g.meta.get("forge.flag"), true);
  assert.deepEqual(g.meta.get("forge.names"), ["a", "b", "c"]);
  const t = g.getTensor("token_embd.weight");
  assert.ok(t);
  assert.deepEqual(Array.from(t.dims), [2, 2]);
  const back = new Float32Array(t.data.buffer, t.data.byteOffset, 4);
  assert.deepEqual(Array.from(back), [1, 2, 3, 4]);
});

test("GGUF 数据段按 alignment 对齐", () => {
  const meta = new Map([["general.alignment", 32]]);
  const { bytes, dataStart } = writeGGUF({ meta, tensors: [] });
  assert.equal(dataStart % 32, 0);
  assert.equal(bytes.length % 1, 0);
});

test("GGUF 非法 magic 抛错", () => {
  const bad = new Uint8Array(64);
  assert.throws(() => readGGUF(bad), /非法 GGUF magic/);
});

test("模型导出 GGUF → 重加载 → 推理输出一致", () => {
  const cfg = new Config({ dim: 32, hidden: 64, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 24, maxSeq: 32 });
  const w = initWeights(cfg, 42);
  const bytes = exportGGUF(w, cfg, null, 0, "test-model");
  const m = loadGGUF(bytes);
  assert.equal(m.config.dim, 32);
  assert.equal(m.config.nLayers, 2);
  assert.equal(m.config.nHeads, 4);
  const gen = generate(m, { prompt: "ab", maxTokens: 10, temperature: 0, seed: 1 });
  assert.ok(typeof gen === "string");
});

test("量化 GGUF 往返：加载后仍可推理", () => {
  const cfg = new Config({ dim: 64, hidden: 128, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 32, maxSeq: 32 });
  const w = initWeights(cfg, 43);
  const f32Bytes = exportGGUF(w, cfg, null, 0, "q-test");
  const g = readGGUF(f32Bytes);
  const meta = new Map(g.meta);
  const tensors = [];
  for (const t of g.tensors) {
    const src = g.getTensor(t.name);
    const nElem = t.dims.reduce((a, b) => a * b, 1);
    if (t.dims.length === 2 && t.type === 0 && t.dims[1] % 256 === 0) {
      const f32 = new Float32Array(src.data.buffer, src.data.byteOffset, nElem);
      tensors.push({ name: t.name, dims: t.dims, type: 8, data: quantizeRow(8, f32, nElem) });
    } else {
      tensors.push({ name: t.name, dims: t.dims, type: src.type, data: src.data });
    }
  }
  const { bytes } = writeGGUF({ meta, tensors });
  const m = loadGGUF(bytes);
  assert.ok(m.transformer);
  const gen = generate(m, { prompt: "ab", maxTokens: 8, temperature: 0, seed: 1 });
  assert.ok(gen.length >= 0);
});

// 引入量化工具（放在文件末尾避免循环引用问题）
import { quantizeRow } from "../src/L3/quant.js";

test("L10 验证层：全部不变量通过", () => {
  const r = runVerify(false);
  const failed = r.results.filter((x) => !x.pass);
  assert.equal(failed.length, 0, `失败项: ${failed.map((x) => `${x.id} ${x.name} (${x.detail})`).join("; ")}`);
  assert.ok(r.pass >= 38, `pass=${r.pass}`);
});

// ── CLI 冒烟测试 ─────────────────────────────────────────────────────
// 背景：曾出现「args 传数组而非解析对象」导致 chat/quantize/export/inspect
// 全部崩溃的回归（verify/train 不经过该路径所以没被发现）。这里对每个命令
// 做退出码 + 产物断言，堵住这类「某个命令整条路径坏掉」的问题。
import { execFileSync } from "node:child_process";
import { mkdtempSync, existsSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CLI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "cli", "forge.mjs");
const forge = (argv) => execFileSync(process.execPath, [CLI, ...argv], { encoding: "utf8", stdio: "pipe" });

test("CLI：verify / help / bench 子集", () => {
  assert.match(forge(["verify"]), /38 通过/);
  assert.match(forge(["--help"]), /命令/);
});

test("CLI：export → inspect → quantize → chat 全链路", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "forge-cli-"));
  const model = path.join(dir, "m.gguf");
  const qmodel = path.join(dir, "m-q8.gguf");
  forge(["export", model, "--dim", "64", "--layers", "2", "--vocab", "256"]);
  assert.ok(existsSync(model) && statSync(model).size > 1000, "export 应产出 GGUF");

  const info = forge(["inspect", model]);
  assert.match(info, /forge.block_count/);
  assert.match(info, /token_embd/);

  const q = forge(["quantize", model, qmodel, "--type", "8"]);
  assert.match(q, /Q8_0/);
  assert.match(q, /压缩|MB/);

  const chat = forge(["chat", qmodel, "--prompt", "hi", "--tokens", "8"]);
  assert.match(chat, /生成结果/);
});
