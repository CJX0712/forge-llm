/**
 * 单元测试 · L0 数值内核 & L1 特殊函数
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  RNG, dot, matvec, gemm, gemmRef, rms, argmax, maxAbsErr, rmse, relL2, cosSim,
} from "../src/L0/numeric.js";
import {
  softmax, logsumexp, logSoftmax, crossEntropy, rmsnorm, layernorm,
  rope, ropeTable, ropeApply, silu, gelu, erf, swiglu, fp16ToFp32, fp32ToFp16,
} from "../src/L1/special.js";

test("RNG 确定性：同种子同序列", () => {
  const a = new RNG(42), b = new RNG(42);
  for (let i = 0; i < 100; i++) assert.equal(a.next(), b.next());
});

test("RNG 不同种子不同序列", () => {
  const a = new RNG(1), b = new RNG(2);
  let same = 0;
  for (let i = 0; i < 20; i++) if (a.next() === b.next()) same++;
  assert.equal(same, 0);
});

test("RNG normal 分布均值≈0 方差≈1", () => {
  const r = new RNG(7);
  const n = 20000;
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) { const v = r.normal(); s += v; s2 += v * v; }
  assert.ok(Math.abs(s / n) < 0.05, `mean=${s / n}`);
  assert.ok(Math.abs(s2 / n - 1) < 0.05, `var=${s2 / n}`);
});

test("dot 与朴素点积一致", () => {
  const r = new RNG(3);
  const a = new Float32Array(100), b = new Float32Array(100);
  for (let i = 0; i < 100; i++) { a[i] = r.normal(); b[i] = r.normal(); }
  let s = 0; for (let i = 0; i < 100; i++) s += a[i] * b[i];
  assert.ok(Math.abs(dot(a, b, 100) - s) < 1e-4);
});

test("gemm 与朴素实现一致（relL2<1e-5）", () => {
  const r = new RNG(5);
  const M = 16, N = 24, K = 20;
  const A = new Float32Array(M * K), B = new Float32Array(N * K);
  for (let i = 0; i < A.length; i++) A[i] = r.normal();
  for (let i = 0; i < B.length; i++) B[i] = r.normal();
  const C = new Float32Array(M * N), Cr = new Float32Array(M * N);
  gemm(C, A, B, M, N, K); gemmRef(Cr, A, B, M, N, K);
  assert.ok(relL2(C, Cr) < 1e-5);
});

test("matvec 与 gemm(M=1) 一致（K≤32 单块时逐位）", () => {
  const r = new RNG(11);
  const N = 32, K = 32;
  const W = new Float32Array(N * K), x = new Float32Array(K);
  for (let i = 0; i < W.length; i++) W[i] = r.normal();
  for (let i = 0; i < K; i++) x[i] = r.normal();
  const o1 = new Float32Array(N), o2 = new Float32Array(N);
  matvec(o1, W, x, N, K);
  gemm(o2, x, W, 1, N, K);
  assert.equal(maxAbsErr(o1, o2), 0);
});

test("matvec 与 gemm(M=1) 在大 K 下数值等价", () => {
  const r = new RNG(12);
  const N = 16, K = 256; // 跨越 gemm 的 BK=32 分块边界，求和顺序不同 → 仅数值等价
  const W = new Float32Array(N * K), x = new Float32Array(K);
  for (let i = 0; i < W.length; i++) W[i] = r.normal();
  for (let i = 0; i < K; i++) x[i] = r.normal();
  const o1 = new Float32Array(N), o2 = new Float32Array(N);
  matvec(o1, W, x, N, K);
  gemm(o2, x, W, 1, N, K);
  assert.ok(relL2(o1, o2) < 1e-6, `relL2=${relL2(o1, o2)}`);
});

test("softmax Σ=1 且非负；logsumexp 恒等", () => {
  const r = new RNG(13);
  const x = new Float32Array(50);
  for (let i = 0; i < 50; i++) x[i] = r.normal() * 10;
  const p = new Float32Array(50);
  softmax(p, x);
  let s = 0; for (const v of p) { assert.ok(v >= 0); s += v; }
  assert.ok(Math.abs(s - 1) < 1e-5);
  let mx = -Infinity; for (const v of x) mx = Math.max(mx, v);
  let ref = 0; for (const v of x) ref += Math.exp(v - mx);
  assert.ok(Math.abs(logsumexp(x) - (mx + Math.log(ref))) < 1e-4);
});

test("softmax 数值稳定：极大 logit 不溢出", () => {
  const x = new Float32Array([1000, 1001, 999]);
  const p = new Float32Array(3);
  softmax(p, x);
  assert.ok(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]));
  assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 1e-5);
});

test("RMSNorm 输出单位 RMS（w=1）", () => {
  const r = new RNG(17);
  const n = 64;
  const x = new Float32Array(n), w = new Float32Array(n).fill(1), out = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = r.normal();
  rmsnorm(out, x, w, n);
  assert.ok(Math.abs(rms(out, n) - 1) < 1e-5);
});

test("RoPE 相对性：内积只依赖位置差", () => {
  const r = new RNG(19);
  const n = 16;
  const q = new Float32Array(n), k = new Float32Array(n);
  for (let i = 0; i < n; i++) { q[i] = r.normal(); k[i] = r.normal(); }
  const qa = new Float32Array(n), kb = new Float32Array(n);
  const qb = new Float32Array(n), kc = new Float32Array(n);
  rope(qa, q, n, 3); rope(kb, k, n, 9);
  rope(qb, q, n, 0); rope(kc, k, n, 6);
  let d1 = 0, d2 = 0;
  for (let i = 0; i < n; i++) { d1 += qa[i] * kb[i]; d2 += qb[i] * kc[i]; }
  assert.ok(Math.abs(d1 - d2) < 1e-4);
});

test("fp16 ↔ fp32 往返误差 < 1e-3 相对", () => {
  const vals = [0, 1, -1, 0.5, 3.14159, -273.15, 1e-4, 65504, 6.1e-5];
  for (const v of vals) {
    const h = fp32ToFp16(v);
    const back = fp16ToFp32(h);
    if (Math.abs(v) > 1e-3) assert.ok(Math.abs(back - v) / Math.abs(v) < 1e-3, `${v} → ${back}`);
  }
});

test("swiglu = silu(gate) ⊙ up", () => {
  const g = new Float32Array([0, 1, -1, 2]);
  const u = new Float32Array([1, 2, 3, 4]);
  const o = new Float32Array(4);
  swiglu(o, g, u, 4);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(o[i] - silu(g[i]) * u[i]) < 1e-6);
});

test("crossEntropy 与手算一致", () => {
  const logits = new Float32Array([1.2, 0.3, -0.5, 2.1]);
  const ref = -Math.log(Math.exp(2.1) / (Math.exp(1.2) + Math.exp(0.3) + Math.exp(-0.5) + Math.exp(2.1)));
  assert.ok(Math.abs(crossEntropy(logits, 3) - ref) < 1e-5);
});

test("gelu 在 0 处等于 0.5x", () => {
  for (const x of [-2, -1, 0, 1, 2]) {
    assert.ok(Math.abs(gelu(x) - 0.5 * x * (1 + erf(x / Math.SQRT2))) < 1e-6);
  }
});

test("ropeTable 与 rope 直接计算一致", () => {
  const tbl = ropeTable(10, 8, 10000);
  const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const a = new Float32Array(8), b = Float32Array.from(x); // ropeApply 原地修改
  rope(a, x, 8, 3, 10000);
  ropeApply(b, 3, tbl, 8);
  assert.ok(maxAbsErr(a, b) < 1e-6, `maxErr=${maxAbsErr(a, b)}`);
});
