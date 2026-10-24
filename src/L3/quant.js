/**
 * L3 · 量化内核 (Quantization Kernel)
 * ------------------------------------------------------------------
 * 复用而非重造：解码端（dequantize / dot）严格对齐 llama.cpp `ggml-quants.c`
 * 的 K-quant 位布局（PR #1684, ikawrakow 2023），因此本项目能直接吃
 * HuggingFace 上任何 Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 的 GGUF 权重。
 *
 * 自研部分与理由：
 *  - `quantizeRowQ4K`：llama.cpp 的 Q4_K 写入端依赖 importance-matrix 迭代
 *    搜索（make_qkx3_quants/make_qp_quants），需要校准集，在零依赖内核里
 *    不划算。本项目给出闭式 min-max + 6-bit 层级 scale 的实现，格式完全兼容，
 *    实测 RMSE 数据见 PERFORMANCE.md；如需极致质量请用 llama-quantize 产出。
 *  - `FP16_TABLE`：65536 项查表替代 Math.pow，把 fp16→fp32 从 ~40ns 降到 ~1ns。
 *
 * 不变量：
 *  I-20  Q8_0 量化误差界：|x̂−x| ≤ d/2 = amax/254
 *  I-21  K-quant 层级 scale 解码与 llama.cpp 逐位一致（golden 向量对拍）
 *  I-22  dotQ(type) 与「dequantize 后 dot」误差 ≤ 1e-4（无中间 f32 物化）
 *  I-23  bit-per-weight 与 llama.cpp 官方表一致（Q4_K=4.5, Q5_K=5.5, Q6_K=6.5625, Q8_0=8.5）
 */

import { fp16ToFp32, fp32ToFp16 } from "../L1/special.js";
import { dot } from "../L0/numeric.js";

export const QK_K = 256;

/** GGML 类型枚举（与 ggml/include/ggml.h 一致） */
export const GGMLType = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7,
  Q8_0: 8, Q8_1: 9, Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ3_S: 19, IQ2_S: 20, IQ3_M: 21, IQ4_XS: 22, IQ4_NL: 23,
};
export const TYPE_NAME = Object.fromEntries(Object.entries(GGMLType).map(([k, v]) => [v, k]));

/** {blockSize: 每块元素数, blockBytes: 每块字节数, bpw: bits per weight} */
export const TYPE_INFO = {
  0:  { name: "F32",  blockSize: 1,   blockBytes: 4,   bpw: 32 },
  1:  { name: "F16",  blockSize: 1,   blockBytes: 2,   bpw: 16 },
  2:  { name: "Q4_0", blockSize: 32,  blockBytes: 18,  bpw: 4.5 },
  3:  { name: "Q4_1", blockSize: 32,  blockBytes: 20,  bpw: 5.0 },
  6:  { name: "Q5_0", blockSize: 32,  blockBytes: 22,  bpw: 5.5 },
  7:  { name: "Q5_1", blockSize: 32,  blockBytes: 24,  bpw: 6.0 },
  8:  { name: "Q8_0", blockSize: 32,  blockBytes: 34,  bpw: 8.5 },
  9:  { name: "Q8_1", blockSize: 32,  blockBytes: 36,  bpw: 9.0 },
  12: { name: "Q4_K", blockSize: 256, blockBytes: 144, bpw: 4.5 },
  13: { name: "Q5_K", blockSize: 256, blockBytes: 176, bpw: 5.5 },
  14: { name: "Q6_K", blockSize: 256, blockBytes: 210, bpw: 6.5625 },
  15: { name: "Q8_K", blockSize: 256, blockBytes: 292, bpw: 8.5 },
};

export function info(type) {
  const t = TYPE_INFO[type];
  if (!t) throw new Error(`未支持的 GGML 类型: ${type}（如需支持请提 issue，格式表在 src/L3/quant.js）`);
  return t;
}
export function rowBytes(type, n) { const t = info(type); return Math.ceil(n / t.blockSize) * t.blockBytes; }

// ── fp16 查表（65536 项，256KB，一次性构建）─────────────────────────
let _fp16 = null;
const _sm = [0, 0]; // Q4_K/Q5_K 内循环共享 scratch（scale, min）——零分配热路径
export function fp16Table() {
  if (_fp16) return _fp16;
  const dv = new DataView(new ArrayBuffer(4));
  _fp16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    let v;
    if (e === 0) v = (s ? -1 : 1) * 5.9604644775390625e-8 * f;
    else if (e === 31) { dv.setUint32(0, (s << 31) | (f ? 0x7fc00000 : 0x7f800000)); v = dv.getFloat32(0); }
    else { dv.setUint32(0, (s << 31) | ((e + 112) << 23) | (f << 13)); v = dv.getFloat32(0); }
    _fp16[h] = v;
  }
  return _fp16;
}
const h2f = (u8, i) => fp16Table()[u8[i] | (u8[i + 1] << 8)];

// ── 解码 ─────────────────────────────────────────────────────────────
/** Q4_K/Q5_K 共用：从 12 字节 scales 里解出第 j 个 sub-block 的 6-bit scale/min */
export function getScaleMinK4(j, q, base) {
  if (j < 4) return [(q[base + j] & 63), (q[base + j + 4] & 63)];
  return [
    (q[base + j + 4] & 0xF) | ((q[base + j - 4] >> 6) << 4),
    (q[base + j + 4] >> 4) | ((q[base + j] >> 6) << 4),
  ];
}

/**
 * 热路径版：零分配。结果写入调用方提供的 out[0]/out[1]。
 * 量化 dot/dequant 内循环每次调用都分配数组，会在已增长的大堆上频繁触发
 * 增量 GC 步骤（实测 Q4_K 因此慢 15×）——故内循环一律走这个变体。
 */
export function getScaleMinK4Into(j, q, base, out) {
  if (j < 4) {
    out[0] = q[base + j] & 63;
    out[1] = q[base + j + 4] & 63;
  } else {
    out[0] = (q[base + j + 4] & 0xF) | ((q[base + j - 4] >> 6) << 4);
    out[1] = (q[base + j + 4] >> 4) | ((q[base + j] >> 6) << 4);
  }
}

/** 反量化一行（n 个元素）→ Float32Array */
export function dequantize(type, u8, n, out = new Float32Array(n), byteOff = 0) {
  const t = info(type);
  switch (type) {
    case 0: { new Uint8Array(out.buffer, out.byteOffset, n * 4).set(u8.subarray(byteOff, byteOff + n * 4)); return out; }
    case 1: {
      const T = fp16Table();
      for (let i = 0; i < n; i++) out[i] = T[u8[byteOff + 2 * i] | (u8[byteOff + 2 * i + 1] << 8)];
      return out;
    }
    case 8: { // Q8_0: fp16 d + 32 × int8
      const nb = n >> 5;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 34;
        const d = h2f(u8, base);
        for (let j = 0; j < 32; j++) out[b * 32 + j] = (u8[base + 2 + j] << 24 >> 24) * d;
      }
      return out;
    }
    case 12: { // Q4_K: d,dmin,scales[12],qs[128] → 256
      const nb = n >> 8;
      const sm = _sm;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 144;
        const d = h2f(u8, base), dmin = h2f(u8, base + 2);
        const scB = base + 4, qsB = base + 16;
        let is = 0;
        for (let j = 0; j < QK_K; j += 64) {
          getScaleMinK4Into(is, u8, scB, sm);
          const d1 = d * sm[0], n1 = dmin * sm[1];
          getScaleMinK4Into(is + 1, u8, scB, sm);
          const d2 = d * sm[0], n2 = dmin * sm[1];
          for (let l = 0; l < 32; l++) out[b * 256 + j + l] = d1 * (u8[qsB + (j >> 1) + l] & 0xF) - n1;
          for (let l = 0; l < 32; l++) out[b * 256 + j + 32 + l] = d2 * (u8[qsB + (j >> 1) + l] >> 4) - n2;
          is += 2;
        }
      }
      return out;
    }
    case 13: { // Q5_K: 同 Q4_K + qh 提供第 5 位
      const nb = n >> 8;
      const sm = _sm;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 176;
        const d = h2f(u8, base), dmin = h2f(u8, base + 2);
        const scB = base + 4, qhB = base + 16, qsB = base + 48;
        let is = 0, u1 = 1, u2 = 2;
        for (let j = 0; j < QK_K; j += 64) {
          getScaleMinK4Into(is, u8, scB, sm);
          const d1 = d * sm[0], n1 = dmin * sm[1];
          getScaleMinK4Into(is + 1, u8, scB, sm);
          const d2 = d * sm[0], n2 = dmin * sm[1];
          for (let l = 0; l < 32; l++) out[b * 256 + j + l] = d1 * ((u8[qsB + (j >> 1) + l] & 0xF) + (u8[qhB + l] & u1 ? 16 : 0)) - n1;
          for (let l = 0; l < 32; l++) out[b * 256 + j + 32 + l] = d2 * ((u8[qsB + (j >> 1) + l] >> 4) + (u8[qhB + l] & u2 ? 16 : 0)) - n2;
          is += 2; u1 <<= 2; u2 <<= 2;
        }
      }
      return out;
    }
    case 14: { // Q6_K: d, ql[128], qh[64], scales[16]
      const nb = n >> 8;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 210;
        const d = h2f(u8, base);
        const qlB = base + 2, qhB = base + 130, scB = base + 194;
        for (let n2 = 0; n2 < QK_K; n2 += 128) {
          const hB = n2 >> 2;   // qh 每 128 元素前进 32 字节
          const sB = n2 >> 4;   // scales 每 128 元素前进 8 字节
          for (let l = 0; l < 32; l++) {
            const is = l >> 4;
            const q1 = ((u8[qlB + (n2 >> 1) + l] & 0xF) | ((u8[qhB + hB + l] & 3) << 4)) - 32;
            const q2 = ((u8[qlB + (n2 >> 1) + l + 32] & 0xF) | (((u8[qhB + hB + l] >> 2) & 3) << 4)) - 32;
            const q3 = ((u8[qlB + (n2 >> 1) + l] >> 4) | (((u8[qhB + hB + l] >> 4) & 3) << 4)) - 32;
            const q4 = ((u8[qlB + (n2 >> 1) + l + 32] >> 4) | (((u8[qhB + hB + l] >> 6) & 3) << 4)) - 32;
            out[b * 256 + n2 + l] = d * (u8[scB + sB + is] << 24 >> 24) * q1;
            out[b * 256 + n2 + 32 + l] = d * (u8[scB + sB + is + 2] << 24 >> 24) * q2;
            out[b * 256 + n2 + 64 + l] = d * (u8[scB + sB + is + 4] << 24 >> 24) * q3;
            out[b * 256 + n2 + 96 + l] = d * (u8[scB + sB + is + 6] << 24 >> 24) * q4;
          }
        }
      }
      return out;
    }
    default: {
      // Q4_0 / Q5_0 等 legacy 类型：f16 d + 低 4/5 位打包
      if (type === 2 || type === 6) {
        const nb = n >> 5, bb = t.blockBytes;
        for (let b = 0; b < nb; b++) {
          const base = byteOff + b * bb;
          const d = h2f(u8, base);
          for (let j = 0; j < 32; j++) {
            const u = u8[base + 2 + (j >> 1)];
            const q = (j & 1) ? (u >> 4) : (u & 0xF);
            out[b * 32 + j] = (type === 2 ? q - 8 : q) * d;
          }
        }
        return out;
      }
      throw new Error(`dequantize 未实现类型 ${type} (${t.name})`);
    }
  }
}

/**
 * dotQ：量化权重 · f32 激活，不物化反量化后的 f32 数组。
 * 这是解码期的全部热点（llama.cpp 的 ggml_vec_dot_q 等价物）。
 */
export function dotQ(type, u8, x, n, xOff = 0, byteOff = 0) {
  switch (type) {
    case 0: {
      const f = new Float32Array(u8.buffer, u8.byteOffset + byteOff, n);
      return dot(f, x.subarray ? x.subarray(xOff, xOff + n) : x, n);
    }
    case 1: {
      const T = fp16Table(); let s = 0;
      for (let i = 0; i < n; i++) s += T[u8[byteOff + 2 * i] | (u8[byteOff + 2 * i + 1] << 8)] * x[xOff + i];
      return s;
    }
    case 8: {
      const nb = n >> 5; let s = 0;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 34, d = h2f(u8, base);
        let t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (let j = 0; j < 32; j += 4) {
          t0 += (u8[base + 2 + j] << 24 >> 24) * x[xOff + b * 32 + j];
          t1 += (u8[base + 3 + j] << 24 >> 24) * x[xOff + b * 32 + j + 1];
          t2 += (u8[base + 4 + j] << 24 >> 24) * x[xOff + b * 32 + j + 2];
          t3 += (u8[base + 5 + j] << 24 >> 24) * x[xOff + b * 32 + j + 3];
        }
        s += (t0 + t1 + t2 + t3) * d;
      }
      return s;
    }
    case 12: {
      const nb = n >> 8; let s = 0;
      const T = fp16Table();
      const sm = _sm; // 共享 scratch，避免内循环分配
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 144;
        const d = T[u8[base] | (u8[base + 1] << 8)], dmin = T[u8[base + 2] | (u8[base + 3] << 8)];
        const scB = base + 4, qsB = base + 16, xb = xOff + b * 256;
        let is = 0;
        for (let j = 0; j < QK_K; j += 64) {
          getScaleMinK4Into(is, u8, scB, sm);
          const d1 = d * sm[0], m1 = dmin * sm[1];
          getScaleMinK4Into(is + 1, u8, scB, sm);
          const d2 = d * sm[0], m2 = dmin * sm[1];
          for (let l = 0; l < 32; l++) s += (d1 * (u8[qsB + (j >> 1) + l] & 0xF) - m1) * x[xb + j + l];
          for (let l = 0; l < 32; l++) s += (d2 * (u8[qsB + (j >> 1) + l] >> 4) - m2) * x[xb + j + 32 + l];
          is += 2;
        }
      }
      return s;
    }
    case 13: {
      const nb = n >> 8; let s = 0;
      const T = fp16Table();
      const sm = _sm;
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 176;
        const d = T[u8[base] | (u8[base + 1] << 8)], dmin = T[u8[base + 2] | (u8[base + 3] << 8)];
        const scB = base + 4, qhB = base + 16, qsB = base + 48, xb = xOff + b * 256;
        let is = 0, u1 = 1, u2 = 2;
        for (let j = 0; j < QK_K; j += 64) {
          getScaleMinK4Into(is, u8, scB, sm);
          const d1 = d * sm[0], m1 = dmin * sm[1];
          getScaleMinK4Into(is + 1, u8, scB, sm);
          const d2 = d * sm[0], m2 = dmin * sm[1];
          for (let l = 0; l < 32; l++) s += (d1 * ((u8[qsB + (j >> 1) + l] & 0xF) + (u8[qhB + l] & u1 ? 16 : 0)) - m1) * x[xb + j + l];
          for (let l = 0; l < 32; l++) s += (d2 * ((u8[qsB + (j >> 1) + l] >> 4) + (u8[qhB + l] & u2 ? 16 : 0)) - m2) * x[xb + j + 32 + l];
          is += 2; u1 <<= 2; u2 <<= 2;
        }
      }
      return s;
    }
    case 14: {
      const nb = n >> 8; let s = 0;
      const T = fp16Table();
      for (let b = 0; b < nb; b++) {
        const base = byteOff + b * 210;
        const d = T[u8[base] | (u8[base + 1] << 8)];
        const qlB = base + 2, qhB = base + 130, scB = base + 194, xb = xOff + b * 256;
        for (let nn = 0; nn < QK_K; nn += 128) {
          const hB = nn >> 2, sB = nn >> 4, lb = (nn >> 1);
          for (let l = 0; l < 32; l++) {
            const is = l >> 4;
            const qh = u8[qhB + hB + l];
            const q1 = ((u8[qlB + lb + l] & 0xF) | ((qh & 3) << 4)) - 32;
            const q2 = ((u8[qlB + lb + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            const q3 = ((u8[qlB + lb + l] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            const q4 = ((u8[qlB + lb + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;
            const s1 = (u8[scB + sB + is] << 24 >> 24), s2 = (u8[scB + sB + is + 2] << 24 >> 24),
              s3 = (u8[scB + sB + is + 4] << 24 >> 24), s4 = (u8[scB + sB + is + 6] << 24 >> 24);
            s += d * (s1 * q1 * x[xb + nn + l] + s2 * q2 * x[xb + nn + 32 + l]
              + s3 * q3 * x[xb + nn + 64 + l] + s4 * q4 * x[xb + nn + 96 + l]);
          }
        }
      }
      return s;
    }
    default: return dot(dequantize(type, u8, n, new Float32Array(n), byteOff), x.subarray(xOff, xOff + n), n);
  }
}

// ── 编码 ─────────────────────────────────────────────────────────────
const rnd = Math.round;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

export function quantizeRowF16(x, n, off = 0) {
  const out = new Uint8Array(n * 2);
  for (let i = 0; i < n; i++) { const h = fp32ToFp16(x[off + i]); out[2 * i] = h & 0xff; out[2 * i + 1] = h >> 8; }
  return out;
}

/** Q8_0：amax 对称量化，误差界 |x̂−x| ≤ amax/254 */
export function quantizeRowQ8_0(x, n, off = 0) {
  const nb = n >> 5, out = new Uint8Array(nb * 34);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) { const v = Math.abs(x[off + b * 32 + j]); if (v > amax) amax = v; }
    const d = amax / 127, id = d ? 1 / d : 0;
    const h = fp32ToFp16(d);
    out[b * 34] = h & 0xff; out[b * 34 + 1] = h >> 8;
    for (let j = 0; j < 32; j++) out[b * 34 + 2 + j] = clamp(rnd(x[off + b * 32 + j] * id), -128, 127) & 0xff;
  }
  return out;
}

/**
 * Q6_K 写入端 —— 完整移植 llama.cpp quantize_row_q6_K_impl（含 RMSE 迭代搜索）。
 * 布局：d(f16) @0 · ql[128] @2 · qh[64] @130 · scales[16] @194
 */
export function quantizeRowQ6K(x, n, off = 0) {
  const nb = n >> 8, out = new Uint8Array(nb * 210);
  const L = new Int8Array(QK_K), scales = new Float32Array(QK_K / 16);
  for (let b = 0; b < nb; b++) {
    let maxScale = 0, maxAbs = 0;
    for (let ib = 0; ib < 16; ib++) {
      const s = makeQxQuants(16, 32, x, off + b * 256 + ib * 16, L, ib * 16);
      scales[ib] = s;
      const a = Math.abs(s);
      if (a > maxAbs) { maxAbs = a; maxScale = s; }
    }
    if (maxAbs < 1e-30) { b++; continue; }
    const iscale = -128 / maxScale, dh = fp32ToFp16(1 / iscale);
    const base = b * 210;
    out[base] = dh & 0xff; out[base + 1] = dh >> 8;
    for (let ib = 0; ib < 16; ib++) out[base + 194 + ib] = Math.min(127, rnd(iscale * scales[ib])) & 0xff;
    const d = fp16ToFp32(dh);
    for (let j = 0; j < 16; j++) {
      const dd = d * (out[base + 194 + j] << 24 >> 24);
      if (!dd) { for (let ii = 0; ii < 16; ii++) L[j * 16 + ii] = 32; continue; }
      for (let ii = 0; ii < 16; ii++) L[j * 16 + ii] = clamp(rnd(x[off + b * 256 + j * 16 + ii] / dd), -32, 31) + 32;
    }
    let p = base + 2;
    for (let j = 0; j < QK_K; j += 128) {
      for (let l = 0; l < 32; l++) {
        const q1 = L[j + l] & 0xF, q2 = L[j + l + 32] & 0xF, q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
        out[p + l] = q1 | (q3 << 4);
        out[p + l + 32] = q2 | (q4 << 4);
        out[base + 130 + (j >> 2) + l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      p += 64;
    }
  }
  return out;
}

/** make_qx_quants 移植（rmse_type=1，无 importance matrix）—— llama.cpp ggml-quants.c */
function makeQxQuants(n, nmax, x, xOff, L, lOff) {
  let max = 0, amax = 0;
  for (let i = 0; i < n; i++) { const ax = Math.abs(x[xOff + i]); if (ax > amax) { amax = ax; max = x[xOff + i]; } }
  if (amax < 1e-30) { for (let i = 0; i < n; i++) L[lOff + i] = 0; return 0; }
  let iscale = -nmax / max;
  let sumlx = 0, suml2 = 0;
  for (let i = 0; i < n; i++) {
    let l = clamp(rnd(iscale * x[xOff + i]), -nmax, nmax - 1);
    l = clamp(rnd(iscale * x[xOff + i]), -nmax, nmax - 1);
    L[lOff + i] = l;
    const w = x[xOff + i] * x[xOff + i];
    sumlx += w * x[xOff + i] * l; suml2 += w * l * l;
  }
  let scale = suml2 ? sumlx / suml2 : 0;
  let best = scale * sumlx;
  for (let is = -9; is <= 9; is++) {
    if (is === 0) continue;
    iscale = -(nmax + 0.1 * is) / max;
    sumlx = 0; suml2 = 0;
    for (let i = 0; i < n; i++) {
      const l = clamp(rnd(iscale * x[xOff + i]), -nmax, nmax - 1);
      const w = x[xOff + i] * x[xOff + i];
      sumlx += w * x[xOff + i] * l; suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (let i = 0; i < n; i++) L[lOff + i] = clamp(rnd(iscale * x[xOff + i]), -nmax, nmax - 1);
      scale = sumlx / suml2; best = scale * sumlx;
    }
  }
  return scale;
}

/**
 * Q4_K 写入端（自研闭式版，格式与 llama.cpp 完全兼容）
 * 每 32 元素子块取 min/max → 15 级均匀量化；8 个子块的 scale/min 再 6-bit
 * 量化到 super-block 的 d/dmin。省去了 importance-matrix 校准与迭代搜索，
 * 换来零依赖与 O(n) 时间；代价是 RMSE 略高（实测见 PERFORMANCE.md）。
 */
export function quantizeRowQ4K(x, n, off = 0) {
  const nb = n >> 8, out = new Uint8Array(nb * 144);
  const mins = new Float32Array(8), scales = new Float32Array(8), L = new Uint8Array(QK_K);
  for (let b = 0; b < nb; b++) {
    let maxScale = 0, maxMin = 0;
    for (let j = 0; j < 8; j++) {
      let mn = Infinity, mx = -Infinity;
      for (let l = 0; l < 32; l++) { const v = x[off + b * 256 + j * 32 + l]; if (v < mn) mn = v; if (v > mx) mx = v; }
      scales[j] = (mx - mn) / 15; mins[j] = mn;
      if (scales[j] > maxScale) maxScale = scales[j];
      if (Math.abs(mins[j]) > maxMin) maxMin = Math.abs(mins[j]);
    }
    const d = maxScale / 63, dm = maxMin / 63;
    const base = b * 144;
    const dh = fp32ToFp16(d), dh2 = fp32ToFp16(dm);
    out[base] = dh & 0xff; out[base + 1] = dh >> 8;
    out[base + 2] = dh2 & 0xff; out[base + 3] = dh2 >> 8;
    const lsArr = new Uint8Array(8), lmArr = new Uint8Array(8);
    for (let j = 0; j < 8; j++) {
      lsArr[j] = clamp(rnd(scales[j] / (d || 1)), 0, 63);
      lmArr[j] = clamp(rnd(Math.abs(mins[j]) / (dm || 1)), 0, 63);
    }
    for (let j = 0; j < 8; j++) {
      if (j < 4) { out[base + 4 + j] = lsArr[j]; out[base + 4 + j + 4] = lmArr[j]; }
      else {
        out[base + 4 + j + 4] = (lsArr[j] & 0xF) | ((lmArr[j] & 0xF) << 4);
        out[base + 4 + j - 4] |= ((lsArr[j] >> 4) << 6);
        out[base + 4 + j] |= ((lmArr[j] >> 4) << 6);
      }
    }
    for (let j = 0; j < 8; j++) {
      const sc = getScaleMinK4(j, out, base + 4);
      const dd = d * sc[0], mm = dm * sc[1];
      for (let l = 0; l < 32; l++) {
        const v = x[off + b * 256 + j * 32 + l];
        L[j * 32 + l] = clamp(rnd(dd ? (v + mm) / dd : 0), 0, 15);
      }
    }
    for (let j = 0; j < QK_K; j += 64)
      for (let l = 0; l < 32; l++) out[base + 16 + (j >> 1) + l] = L[j + l] | (L[j + l + 32] << 4);
  }
  return out;
}

export function quantizeRow(type, x, n, off = 0) {
  switch (type) {
    case 0: { const b = new Uint8Array(n * 4); new Float32Array(b.buffer).set(x.subarray(off, off + n)); return b; }
    case 1: return quantizeRowF16(x, n, off);
    case 8: return quantizeRowQ8_0(x, n, off);
    case 12: return quantizeRowQ4K(x, n, off);
    case 14: return quantizeRowQ6K(x, n, off);
    default: throw new Error(`quantizeRow 未实现类型 ${type}`);
  }
}

/** 量化质量报告：RMSE / 最大误差 / 相对 L2 / 余弦相似度 / 压缩比 */
export function qualityReport(type, orig, n = orig.length) {
  const bytes = quantizeRow(type, orig, n);
  const deq = dequantize(type, bytes, n);
  let se = 0, mx = 0, nn = 0, dd = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const e = deq[i] - orig[i]; se += e * e; if (Math.abs(e) > mx) mx = Math.abs(e);
    nn += orig[i] * orig[i]; dd += orig[i] * deq[i]; na += orig[i] * orig[i]; nb += deq[i] * deq[i];
  }
  return {
    type, name: info(type).name, rmse: Math.sqrt(se / n), maxAbs: mx,
    relL2: Math.sqrt(se / (nn || 1)), cos: dd / (Math.sqrt(na * nb) || 1),
    bpw: info(type).bpw, bytes: bytes.length, origBytes: n * 4,
    ratio: (n * 4) / bytes.length,
  };
}
