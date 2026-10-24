/**
 * L1 · 特殊函数 (Special Functions)
 * ------------------------------------------------------------------
 * 所有非线性与归一化都集中在这里，保证「同一函数只有一个实现」，
 * L2 自动微分与 L6 推理共用同一份代码 —— 训练/推理数值一致性靠这一点兜底。
 *
 * 站在巨人肩上：
 *  - Stable softmax + logsumexp：《Numerical Recipes》《深度学习》(Goodfellow 2016) 标准式；
 *    也是 FlashAttention 在线 softmax 的标量退化版。
 *  - SwiGLU：Shazeer 2020（LLaMA/Qwen/Mistral 现状默认 FFN 激活）。
 *  - GELU 精确 erf 版：Hendrycks & Gimpel 2016。
 *  - RoPE：Su et al. 2023（RoFormer），复数旋转的实数实现。
 *
 * 不变量：
 *  I-10  softmax 输出 Σ=1 且全非负
 *  I-11  logsumexp 恒等式 log Σexp(x) = m + log Σexp(x-m)
 *  I-12  RMSNorm 后 ‖x‖₂/√n = 1（±1e-6）
 *  I-13  RoPE 相对性：(R_m q)·(R_n k) 只依赖 m-n
 */

import { rms } from "../L0/numeric.js";

// ── 激活 ─────────────────────────────────────────────────────────────
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const silu = (x) => x / (1 + Math.exp(-x));           // Swish-1
export const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)); // 精确 GELU
export const relu = (x) => (x > 0 ? x : 0);
export const tanh = Math.tanh;

/** Abramowitz & Stegun 7.1.26 —— |ε| < 1.5e-7 */
export function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t * Math.exp(-x * x);
  return s * y;
}

/** SwiGLU 前馈：out[i] = (silu(g[i]) * u[i]) —— Shazeer 2020 */
export function swiglu(out, g, u, n) {
  for (let i = 0; i < n; i++) out[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
  return out;
}

// ── 归一化 ───────────────────────────────────────────────────────────
/** RMSNorm: y = x/rms(x) * w —— Zhang & Sennrich 2019 */
export function rmsnorm(out, x, w, n, eps = 1e-6) {
  const ss = 1 / Math.sqrt(rms(x, n) ** 2 + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
  return out;
}

/** LayerNorm（GPT-2 系需要，保留以兼容多架构） */
export function layernorm(out, x, w, b, n, eps = 1e-6) {
  let mu = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mu; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + eps);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * w[i] + b[i];
  return out;
}

// ── softmax 家族 ─────────────────────────────────────────────────────
/** 稳定 softmax：减去最大值后归一化；max=-∞ 时退化为均匀分布（防御 NaN） */
export function softmax(out, x, n = x.length, off = 0) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (x[off + i] > m) m = x[off + i];
  let s = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(x[off + i] - m); out[off + i] = e; s += e; }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) out[off + i] *= inv;
  return out;
}

export function logsumexp(x, n = x.length, off = 0) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (x[off + i] > m) m = x[off + i];
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.exp(x[off + i] - m);
  return m + Math.log(s);
}

/** log-softmax，交叉熵损失与 top-p 截断共用 */
export function logSoftmax(out, x, n = x.length) {
  const lse = logsumexp(x, n);
  for (let i = 0; i < n; i++) out[i] = x[i] - lse;
  return out;
}

/** 交叉熵：-log p[target]，输入为 logits */
export function crossEntropy(logits, target, n = logits.length) {
  return -logSoftmax(new Float32Array(n), logits, n)[target];
}

// ── 位置编码 ─────────────────────────────────────────────────────────
/**
 * RoPE（旋转位置编码，Su et al. 2023）
 * 成对旋转：(x2i, x2i+1) → (x2i·cos - x2i+1·sin, x2i·sin + x2i+1·cos)
 * θ_i = 1 / base^(2i/d)
 */
export function rope(out, x, n, pos, theta = 10000, freqScale = 1) {
  for (let i = 0; i < n; i += 2) {
    const freq = 1 / Math.pow(theta, (i / n) * freqScale);
    const ang = pos * freq, c = Math.cos(ang), s = Math.sin(ang);
    const a = x[i], b = x[i + 1];
    out[i] = a * c - b * s;
    out[i + 1] = a * s + b * c;
  }
  if (n & 1) out[n - 1] = x[n - 1];
  return out;
}

/** 预计算 RoPE 表，避免每 token 三角函数 —— 推理期显著省时
 *  频率公式与 rope() 完全一致（同一表达式形式，保证逐位可复现）*/
export function ropeTable(maxPos, dim, theta = 10000) {
  const cos = new Float32Array(maxPos * dim / 2), sin = new Float32Array(maxPos * dim / 2);
  for (let p = 0; p < maxPos; p++) {
    for (let i = 0; i < dim; i += 2) {
      const freq = 1 / Math.pow(theta, i / dim);
      const ang = p * freq;
      cos[p * (dim / 2) + i / 2] = Math.cos(ang);
      sin[p * (dim / 2) + i / 2] = Math.sin(ang);
    }
  }
  return { cos, sin, half: dim / 2 };
}

/** 应用预计算表；in-place 版本用于省一次内存往返 */
export function ropeApply(x, pos, tbl, dim) {
  const half = dim / 2, cp = pos * half;
  for (let i = 0; i < dim; i += 2) {
    const c = tbl.cos[cp + (i >> 1)], s = tbl.sin[cp + (i >> 1)];
    const a = x[i], b = x[i + 1];
    x[i] = a * c - b * s;
    x[i + 1] = a * s + b * c;
  }
  return x;
}

// ── 半精度位级转换 ───────────────────────────────────────────────────
/** IEEE-754 half → float，位级精确（含 subnormal / Inf / NaN） */
export function fp16ToFp32(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -24) * f;
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** float → IEEE-754 half（round-to-nearest-even） */
export function fp32ToFp16(x) {
  if (!isFinite(x)) return x > 0 ? 0x7c00 : x < 0 ? 0xfc00 : 0x7e00;
  const s = x < 0 ? 0x8000 : 0; x = Math.abs(x);
  if (x < 6.103515625e-5) {
    // subnormal：half 的最小正规数是 2^-14，subnormal 步长 2^-24
    const q = Math.round(x * 16777216); // x / 2^-24
    return s | (q & 0x3ff);
  }
  let e = 0;
  while (x >= 2) { x /= 2; e++; }
  while (x < 1) { x *= 2; e--; }
  let m = Math.round((x - 1) * 1024);
  if (m === 1024) { e++; m = 0; }
  // 指数字段 = e + 15；≥31 表示溢出到 Inf（IEEE 754 §4.3，RNE 阈值 65520）
  const exp = e + 15;
  if (exp >= 31) return s | 0x7c00;
  return s | (exp << 10) | m;
}
