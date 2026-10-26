/**
 * 单元测试 · L3 量化 & L4 组件 & L5 注意力
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  dequantize, quantizeRow, dotQ, rowBytes, info, qualityReport,
  TYPE_INFO, GGMLType, quantizeRowQ4K, quantizeRowQ5K, quantizeRowQ6K, quantizeRowQ8_0, quantizeRowF16,
} from "../src/L3/quant.js";
import { fp16ToFp32 } from "../src/L1/special.js";
import { QuantLinear, SwiGLU } from "../src/L4/nn.js";
import { attentionNaive, flashAttention, KVCache, PagedKVCache } from "../src/L5/attention.js";
import { RNG, dot, maxAbsErr, cosSim } from "../src/L0/numeric.js";
import { matvec } from "../src/L0/numeric.js";

test("bit-per-weight 与 llama.cpp 官方表一致", () => {
  assert.equal(info(GGMLType.Q4_K).bpw, 4.5);
  assert.equal(info(GGMLType.Q5_K).bpw, 5.5);
  assert.equal(info(GGMLType.Q6_K).bpw, 6.5625);
  assert.equal(info(GGMLType.Q8_0).bpw, 8.5);
  assert.equal(info(GGMLType.F16).bpw, 16);
  assert.equal(info(GGMLType.F32).bpw, 32);
});

test("rowBytes 与块长一致", () => {
  assert.equal(rowBytes(GGMLType.Q4_K, 256), 144);
  assert.equal(rowBytes(GGMLType.Q4_K, 512), 288);
  assert.equal(rowBytes(GGMLType.Q6_K, 256), 210);
  assert.equal(rowBytes(GGMLType.Q8_0, 32), 34);
  assert.equal(rowBytes(GGMLType.Q8_0, 64), 68);
  assert.equal(rowBytes(GGMLType.F32, 100), 400);
});

test("F16 往返精度", () => {
  const n = 64;
  const x = new Float32Array(n);
  const rng = new RNG(1);
  for (let i = 0; i < n; i++) x[i] = rng.normal();
  const q = quantizeRowF16(x, n);
  const dq = dequantize(GGMLType.F16, q, n);
  let maxRel = 0;
  for (let i = 0; i < n; i++) maxRel = Math.max(maxRel, Math.abs(dq[i] - x[i]) / (Math.abs(x[i]) + 1e-3));
  assert.ok(maxRel < 1e-3, `maxRel=${maxRel}`);
});

test("Q8_0 误差界 |x̂−x| ≤ amax/254", () => {
  const n = 256;
  const x = new Float32Array(n);
  const rng = new RNG(2);
  let amax = 0;
  for (let i = 0; i < n; i++) { x[i] = rng.normal(); amax = Math.max(amax, Math.abs(x[i])); }
  const q = quantizeRowQ8_0(x, n);
  const dq = dequantize(GGMLType.Q8_0, q, n);
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(dq[i] - x[i]));
  assert.ok(worst <= amax / 254 + 1e-6, `worst=${worst} bound=${amax / 254}`);
});

test("dotQ 与「反量化后点积」一致（无中间物化）", () => {
  const n = 512;
  const rng = new RNG(3);
  const x = new Float32Array(n), y = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = rng.normal() * 0.8; y[i] = rng.normal(); }
  for (const t of [GGMLType.F16, GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K]) {
    const q = quantizeRow(t, x, n);
    const dq = dequantize(t, q, n);
    const a = dot(dq, y, n);
    const b = dotQ(t, q, y, n);
    assert.ok(Math.abs(a - b) <= 1e-3 * (Math.abs(a) + 1), `${info(t).name}: ${a} vs ${b}`);
  }
});

test("Q5_K 写入端：字节数 / 往返误差 / 5 位取值域", () => {
  const n = 256, rng = new RNG(7);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.normal() * 0.5;
  const q = quantizeRowQ5K(x, n);
  assert.equal(q.length, 176);
  assert.equal(quantizeRow(GGMLType.Q5_K, x, n).length, 176);
  const dq = dequantize(GGMLType.Q5_K, q, n);
  let se = 0, sx = 0;
  for (let i = 0; i < n; i++) { se += (dq[i] - x[i]) ** 2; sx += x[i] ** 2; }
  const rel = Math.sqrt(se / sx);
  assert.ok(rel < 0.05, `Q5_K 相对 RMSE=${rel}`);
  // 5 位：qs 低 4 位 + qh 第 5 位；qh 的 32 字节必须被用到（否则等于退化成 Q4_K）
  let qhUsed = 0;
  for (let i = 0; i < 32; i++) if (q[16 + i] !== 0) qhUsed++;
  assert.ok(qhUsed > 0, "qh 全零 ⇒ 第 5 位未写入");
});

test("Q6_K 位布局：super-block 的 d 在块尾（ql/qh/scales 与整体缩放无关）", () => {
  // 判据：K-quant 的 ql/qh/scales 对正标量缩放不变，只有 d 随缩放线性变化。
  // 若把 d 误放在块头（Q4_K/Q5_K 的写法），缩放后变化的就会是前两字节而非末两字节。
  const n = 256, rng = new RNG(8);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.normal();
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = x[i] * 4;
  const qx = quantizeRowQ6K(x, n), qy = quantizeRowQ6K(y, n);
  assert.equal(qx.length, 210);
  let headSame = true, tailDiff = false;
  for (let b = 0; b < n / 256; b++) {
    for (let i = 0; i < 208; i++) if (qx[b * 210 + i] !== qy[b * 210 + i]) headSame = false;
    if (qx[b * 210 + 208] !== qy[b * 210 + 208] || qx[b * 210 + 209] !== qy[b * 210 + 209]) tailDiff = true;
  }
  assert.ok(headSame, "ql/qh/scales（块内前 208 字节）应随整体缩放不变");
  assert.ok(tailDiff, "d（块尾 2 字节）应随整体缩放变化");
  const dx = fp16ToFp32(qx[208] | (qx[209] << 8));
  const dy = fp16ToFp32(qy[208] | (qy[209] << 8));
  assert.ok(Math.abs(dy / dx - 4) < 0.02, `d 应线性随缩放：${dy} / ${dx} = ${dy / dx}`);
});

test("量化质量报告：压缩比与余弦", () => {
  const n = 1024;
  const rng = new RNG(4);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.normal() * 0.6;
  const q8 = qualityReport(GGMLType.Q8_0, x, n);
  const q4 = qualityReport(GGMLType.Q4_K, x, n);
  assert.ok(q8.ratio > 3.5 && q8.ratio < 4.0, `Q8_0 ratio=${q8.ratio}`);
  assert.ok(q4.ratio > 6.5 && q4.ratio < 7.5, `Q4_K ratio=${q4.ratio}`);
  assert.ok(q8.cos > 0.999, `Q8_0 cos=${q8.cos}`);
  assert.ok(q4.cos > 0.995, `Q4_K cos=${q4.cos}`);
});

test("QuantLinear：量化输出与 f32 输出高度一致", () => {
  const N = 64, K = 128;
  const rng = new RNG(5);
  const W = new Float32Array(N * K);
  for (let i = 0; i < W.length; i++) W[i] = rng.normal();
  const x = new Float32Array(K);
  for (let i = 0; i < K; i++) x[i] = rng.normal();
  const f32 = QuantLinear.fromF32(W, N, K, 0);
  const q8 = QuantLinear.fromF32(W, N, K, GGMLType.Q8_0);
  const o1 = new Float32Array(N), o2 = new Float32Array(N);
  f32.forward(x, o1); q8.forward(x, o2);
  assert.ok(cosSim(o1, o2) > 0.999, `cos=${cosSim(o1, o2)}`);
});

test("QuantLinear.toF32 与 dequantize 一致", () => {
  const N = 32, K = 64;
  const rng = new RNG(6);
  const W = new Float32Array(N * K);
  for (let i = 0; i < W.length; i++) W[i] = rng.normal();
  const ql = QuantLinear.fromF32(W, N, K, GGMLType.Q8_0);
  const deq = dequantize(GGMLType.Q8_0, ql.bytes, N * K);
  assert.ok(maxAbsErr(ql.toF32(), deq) === 0);
});

test("SwiGLU 块 == 手动分解", () => {
  const dim = 32, hidden = 64;
  const rng = new RNG(7);
  const gW = new Float32Array(hidden * dim), uW = new Float32Array(hidden * dim), dW = new Float32Array(dim * hidden);
  for (let i = 0; i < gW.length; i++) { gW[i] = rng.normal(); uW[i] = rng.normal(); }
  for (let i = 0; i < dW.length; i++) dW[i] = rng.normal();
  const x = new Float32Array(dim);
  for (let i = 0; i < dim; i++) x[i] = rng.normal();
  const sw = new SwiGLU(
    QuantLinear.fromF32(gW, hidden, dim, 0),
    QuantLinear.fromF32(uW, hidden, dim, 0),
    QuantLinear.fromF32(dW, dim, hidden, 0), dim, hidden);
  const out = new Float32Array(dim);
  sw.forward(x, out);
  const gv = new Float32Array(hidden), uv = new Float32Array(hidden), act = new Float32Array(hidden);
  matvec(gv, gW, x, hidden, dim);
  matvec(uv, uW, x, hidden, dim);
  for (let i = 0; i < hidden; i++) act[i] = (gv[i] / (1 + Math.exp(-gv[i]))) * uv[i];
  const ref = new Float32Array(dim);
  matvec(ref, dW, act, dim, hidden);
  assert.ok(maxAbsErr(out, ref) < 1e-5);
});

test("flashAttention == 朴素注意力", () => {
  const n = 64, d = 16;
  const rng = new RNG(8);
  const q = new Float32Array(d);
  for (let i = 0; i < d; i++) q[i] = rng.normal();
  const K = new Float32Array(n * d), V = new Float32Array(n * d);
  for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
  const o1 = new Float32Array(d), o2 = new Float32Array(d);
  flashAttention(o1, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
  attentionNaive(o2, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
  assert.ok(maxAbsErr(o1, o2) < 1e-4, `maxErr=${maxAbsErr(o1, o2)}`);
});

test("因果掩码：pos 之后的注意力权重恒为 0", () => {
  const n = 16, d = 8;
  const rng = new RNG(9);
  const q = new Float32Array(d);
  for (let i = 0; i < d; i++) q[i] = rng.normal();
  const K = new Float32Array(n * d), V = new Float32Array(n * d);
  for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
  const pos = 5;
  const o = new Float32Array(d);
  flashAttention(o, q, K, V, n, d, 1 / Math.sqrt(d), true, pos, 0);
  // 与「只喂前 pos+1 个」的朴素结果一致 → 后续 token 未参与
  const oRef = new Float32Array(d);
  attentionNaive(oRef, q, K, V, pos + 1, d, 1 / Math.sqrt(d), true, pos, 0);
  assert.ok(maxAbsErr(o, oRef) < 1e-5);
});

test("KVCache 增量解码 == 全量重算", () => {
  const n = 32, d = 8;
  const rng = new RNG(10);
  const K = new Float32Array(n * d), V = new Float32Array(n * d);
  for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
  const q = new Float32Array(d);
  for (let i = 0; i < d; i++) q[i] = rng.normal();
  const cache = new KVCache(1, n, d);
  for (let p = 0; p < n; p++) cache.put(0, p, K.subarray(p * d, p * d + d), V.subarray(p * d, p * d + d));
  const oCache = new Float32Array(d);
  flashAttention(oCache, q, cache.kView(0, n), cache.vView(0, n), n, d, 1 / Math.sqrt(d), true, n - 1, 0);
  const oFull = new Float32Array(d);
  flashAttention(oFull, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
  assert.ok(maxAbsErr(oCache, oFull) < 1e-5);
});

test("PagedKVCache 与连续 KVCache 内容一致", () => {
  const n = 24, d = 8, blockSize = 8;
  const rng = new RNG(11);
  const K = new Float32Array(n * d), V = new Float32Array(n * d);
  for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
  const cont = new KVCache(1, n, d);
  const paged = new PagedKVCache(1, n, d, blockSize);
  for (let p = 0; p < n; p++) {
    const k = K.subarray(p * d, p * d + d), v = V.subarray(p * d, p * d + d);
    cont.put(0, p, k, v);
    paged.put(0, p, k, v);
  }
  for (let p = 0; p < n; p++) {
    assert.ok(maxAbsErr(cont.sliceK(0, p), paged.sliceK(0, p)) === 0);
    assert.ok(maxAbsErr(cont.sliceV(0, p), paged.sliceV(0, p)) === 0);
  }
});
