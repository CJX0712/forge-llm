/**
 * dump_stages.mjs —— 导出 forge 前向的逐阶段中间量，供 cmp_full.py 逐点对拍
 *
 * 与 verify_engine.mjs 的分工：
 *   verify_engine.mjs 走「出厂代码路径」(Transformer.forward + KVCache + gatherKV)，
 *                      用于证明真实发布代码正确；
 *   dump_stages.mjs   手工展开每一层每一阶段并落盘，用于**定位**误差出现在哪一环
 *                      （embed / afterAttn / afterFFN / xFinal / logits）。
 *
 * 用法（在仓库根目录）：node tools/interop/dump_stages.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { loadGGUF } from "../../src/L9/engine.js";
import { rmsnorm, ropeTable } from "../../src/L1/special.js";
import { attentionDecode } from "../../src/L5/attention.js";
import { addInto, argmax } from "../../src/L0/numeric.js";

const m = loadGGUF(new Uint8Array(readFileSync("tools/interop/out/llama-official.gguf")));
const tr = m.transformer, c = m.config;
const dHead = c.dHead, qD = c.nHeads * dHead, kvD = c.nKvHeads * dHead;
const rope = ropeTable(c.maxSeq, dHead, c.ropeTheta);
const ids = [2, 3, 4, 5];

const x = new Float32Array(c.dim);
const xb = new Float32Array(c.dim);
const xb2 = new Float32Array(c.dim);
const q = new Float32Array(c.qkvOut);
const att = new Float32Array(c.dim);
const tmpK = new Float32Array(c.maxSeq * dHead);
const tmpV = new Float32Array(c.maxSeq * dHead);
const scores = new Float32Array(c.maxSeq);
const KC = Array.from({ length: c.nLayers }, () => []);
const VC = Array.from({ length: c.nLayers }, () => []);
const out = {};

/** 给某个 head 的子向量原地施加 RoPE（与 src/L6/transformer.js 的 applyRopeHead 同式） */
const applyRope = (xx, base, d, pos, tbl) => {
  const half = d >> 1, cp = pos * half;
  for (let i = 0; i < d; i += 2) {
    const cc = tbl.cos[cp + (i >> 1)], ss = tbl.sin[cp + (i >> 1)];
    const a = xx[base + i], b = xx[base + i + 1];
    xx[base + i] = a * cc - b * ss;
    xx[base + i + 1] = a * ss + b * cc;
  }
};

for (let pos = 0; pos < ids.length; pos++) {
  tr._embed(ids[pos], x);
  if (pos === 0) out.embed = Array.from(x);
  for (let l = 0; l < c.nLayers; l++) {
    const L = tr.layers[l];
    rmsnorm(xb, x, L.ln1, c.dim, c.normEps);
    L.wqkv.forward(xb, q);
    const qh = q.subarray(0, qD);
    const kh = q.subarray(qD, qD + kvD);
    const vh = q.subarray(qD + kvD, qD + 2 * kvD);
    for (let h = 0; h < c.nHeads; h++) applyRope(qh, h * dHead, dHead, pos, rope);
    for (let h = 0; h < c.nKvHeads; h++) applyRope(kh, h * dHead, dHead, pos, rope);
    KC[l].push(Float32Array.from(kh));
    VC[l].push(Float32Array.from(vh));
    att.fill(0, 0, c.dim);
    for (let h = 0; h < c.nHeads; h++) {
      const kvh = Math.floor(h / c.groupSize);
      for (let p = 0; p <= pos; p++) {
        for (let i = 0; i < dHead; i++) {
          tmpK[p * dHead + i] = KC[l][p][kvh * dHead + i];
          tmpV[p * dHead + i] = VC[l][p][kvh * dHead + i];
        }
      }
      attentionDecode(
        att.subarray(h * dHead, (h + 1) * dHead),
        qh.subarray(h * dHead, (h + 1) * dHead),
        tmpK, tmpV, pos + 1, dHead, 1 / Math.sqrt(dHead), true, pos, 0, scores
      );
    }
    L.wo.forward(att, xb2);
    addInto(x, xb2, c.dim);
    out[`P${pos}_L${l}_afterAttn`] = Array.from(x);
    rmsnorm(xb, x, L.ln2, c.dim, c.normEps);
    L.ffn.forward(xb, xb2);
    addInto(x, xb2, c.dim);
    out[`P${pos}_L${l}_afterFFN`] = Array.from(x);
  }
}

rmsnorm(xb, x, tr.lnF, c.dim, c.normEps);
out.xFinal = Array.from(xb);
const logits = new Float32Array(c.vocabSize);
tr.wcls.forward(xb, logits);
out.logits = Array.from(logits);

writeFileSync("tools/interop/out/js-full.json", JSON.stringify(out));
console.log("逐阶段 dump → tools/interop/out/js-full.json");
console.log("手工 forward logits[:6] =", Array.from(logits.slice(0, 6)).map((v) => v.toFixed(5)).join(", "));
console.log("argmax =", argmax(logits, c.vocabSize));
