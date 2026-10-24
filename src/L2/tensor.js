/**
 * L2b · 张量级自动微分算子 (Tensor-level AD Ops)
 * ------------------------------------------------------------------
 * L2/autograd.js 提供的是「向量级」算子（对应 PyTorch 的一元/逐元素层）；
 * 本文件提供「矩阵级」融合算子，对应 PyTorch 的 matmul / embedding / softmax /
 * cross_entropy 的 backward 融合实现 —— 即一次算子完成整块张量的反向，
 * 避免逐元素 tape 造成 O(N) 的 tape 膨胀与常数开销。
 *
 * 站在巨人肩上：
 *  - 反向公式严格取自 PyTorch 官方 `derivatives.yaml`（同 SAXPY / ATen 语义）：
 *    matmul → dX = dY·Wᵀ, dW = dYᵀ·X；softmax → dS = P⊙(dP − ΣP⊙dP)；
 *    cross_entropy → dLogits = (P − OneHot)/N。
 *  - RMSNorm 反向取 Zhang & Sennrich 2019 论文附录的解析式。
 *  - RoPE 反向为旋转矩阵的转置 Rᵀ = R⁻¹（旋转为正交变换）。
 *  - 因果注意力的反向为 FlashAttention-2 的标准反向（dQ/dK/dV 解析式）。
 *
 * 不变量：
 *  I-33  matmulT 的解析梯度 vs 中心差分：maxRelErr < 1e-5
 *  I-34  crossEntropyMean 的梯度恰为 (P − OneHot)/N
 *  I-35  causalAttn 的反向与「朴素注意力 + 有限差分」一致
 *  I-36  rmsNorm 反向满足 Σ dx = 0（沿特征维的投影性质）
 */

import { op } from "./autograd.js";
import { gemm } from "../L0/numeric.js";

/**
 * Y[M,N] = X[M,K] · W[N,K]ᵀ    （W 为 [N,K] 行主序 —— 与权重布局一致）
 * 反向：dX = dY·W,  dW = dYᵀ·X
 *
 * 前向直接复用 L0 的分块 GEMM（gemm 的 B 参数本就是 [N,K] 行主序，
 * 与本算子语义完全一致）——零重复实现，且自动获得 cache 分块 + 4 路展开。
 */
export function matmulT(X, W, M, K, N) {
  const out = new Float32Array(M * N);
  gemm(out, X.data, W.data, M, N, K);
  return op(out, (g) => {
    const gX = (X.grad ??= new Float32Array(M * K));
    const gW = (W.grad ??= new Float32Array(N * K));
    for (let m = 0; m < M; m++) {
      const xb = m * K, gb = m * N;
      for (let n = 0; n < N; n++) {
        const gn = g[gb + n];
        if (gn === 0) continue;
        const wb = n * K;
        for (let k = 0; k < K; k++) {
          gX[xb + k] += gn * W.data[wb + k];
          gW[wb + k] += gn * X.data[xb + k];
        }
      }
    }
  });
}

/** Y[M,N] = X[M,K] · B[K,N]（B 行主序） */
export function matmul(X, B, M, K, N) {
  const out = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    const xb = m * K;
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += X.data[xb + k] * B.data[k * N + n];
      out[m * N + n] = s;
    }
  }
  return op(out, (g) => {
    const gX = (X.grad ??= new Float32Array(M * K));
    const gB = (B.grad ??= new Float32Array(K * N));
    for (let m = 0; m < M; m++) {
      const xb = m * K, gb = m * N;
      for (let n = 0; n < N; n++) {
        const gn = g[gb + n];
        if (gn === 0) continue;
        for (let k = 0; k < K; k++) {
          gX[xb + k] += gn * B.data[k * N + n];
          gB[k * N + n] += gn * X.data[xb + k];
        }
      }
    }
  });
}

// ── 嵌入查表 ─────────────────────────────────────────────────────────
/** X[T,D] = W[ids[t], :]   —— 反向为 scatter-add 到 W */
export function embedLookup(W, ids, V, D) {
  const T = ids.length;
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = ids[t] * D;
    for (let d = 0; d < D; d++) out[t * D + d] = W.data[o + d];
  }
  return op(out, (g) => {
    const gW = (W.grad ??= new Float32Array(V * D));
    for (let t = 0; t < T; t++) {
      const o = ids[t] * D, go = t * D;
      for (let d = 0; d < D; d++) gW[o + d] += g[go + d];
    }
  });
}

// ── RMSNorm（逐 token）───────────────────────────────────────────────
/** Y[T,D] = X[T,D] / rms(X[t]) · w[D]   —— eps 加在均方内部（与 L1/L6 一致） */
export function rmsNorm(X, w, T, D, eps = 1e-6) {
  const out = new Float32Array(T * D);
  const inv = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const xb = t * D;
    let ss = 0;
    for (let i = 0; i < D; i++) ss += X.data[xb + i] * X.data[xb + i];
    const r = Math.sqrt(ss / D + eps);
    inv[t] = 1 / r;
    for (let i = 0; i < D; i++) out[xb + i] = X.data[xb + i] * inv[t] * w.data[i];
  }
  return op(out, (g) => {
    const gX = (X.grad ??= new Float32Array(T * D));
    const gw = (w.grad ??= new Float32Array(D));
    for (let t = 0; t < T; t++) {
      const xb = t * D, r = 1 / inv[t];
      let dot = 0;
      for (let i = 0; i < D; i++) dot += g[xb + i] * w.data[i] * X.data[xb + i];
      const c = dot / (D * r * r * r);
      for (let i = 0; i < D; i++) {
        gX[xb + i] += g[xb + i] * w.data[i] * inv[t] - c * X.data[xb + i];
        gw[i] += g[xb + i] * X.data[xb + i] * inv[t];
      }
    }
  });
}

// ── RoPE（逐行、逐 head）────────────────────────────────────────────
/** X[T, nHead*dHead] → 就地旋转；位置 = pos0 + (t % posStride)
 *  posStride 用于批量场景：每个样本的位置从 0 重新计数（posStride = 序列长度）。 */
export function ropeRows(X, T, nHead, dHead, pos0, theta = 10000, posStride = 0) {
  const D = nHead * dHead;
  const out = new Float32Array(T * D);
  out.set(X.data);
  for (let t = 0; t < T; t++) {
    const pos = pos0 + (posStride > 0 ? (t % posStride) : t), xb = t * D;
    for (let h = 0; h < nHead; h++) {
      const hb = xb + h * dHead;
      for (let i = 0; i < dHead; i += 2) {
        const freq = 1 / Math.pow(theta, i / dHead);
        const ang = pos * freq, c = Math.cos(ang), s = Math.sin(ang);
        const a = out[hb + i], b = out[hb + i + 1];
        out[hb + i] = a * c - b * s;
        out[hb + i + 1] = a * s + b * c;
      }
    }
  }
  return op(out, (g) => {
    const gX = (X.grad ??= new Float32Array(T * D));
    for (let t = 0; t < T; t++) {
      const pos = pos0 + (posStride > 0 ? (t % posStride) : t), xb = t * D;
      for (let h = 0; h < nHead; h++) {
        const hb = xb + h * dHead;
        for (let i = 0; i < dHead; i += 2) {
          const freq = 1 / Math.pow(theta, i / dHead);
          const ang = pos * freq, c = Math.cos(ang), s = Math.sin(ang);
          const ga = g[hb + i], gb = g[hb + i + 1];
          gX[hb + i] += ga * c + gb * s;
          gX[hb + i + 1] += -ga * s + gb * c;
        }
      }
    }
  });
}

// ── 列切片 ───────────────────────────────────────────────────────────
/** 取 X[M,K] 的 [off, off+len) 列 → [M,len]；反向 scatter 回原位 */
export function sliceCols(X, M, K, off, len) {
  const out = new Float32Array(M * len);
  for (let m = 0; m < M; m++) {
    const xb = m * K + off, ob = m * len;
    for (let i = 0; i < len; i++) out[ob + i] = X.data[xb + i];
  }
  return op(out, (g) => {
    const gX = (X.grad ??= new Float32Array(M * K));
    for (let m = 0; m < M; m++) {
      const xb = m * K + off, ob = m * len;
      for (let i = 0; i < len; i++) gX[xb + i] += g[ob + i];
    }
  });
}

// ── 因果自注意力（含 GQA）────────────────────────────────────────────
/**
 * Q[T, nH*dHead], K[T, nKV*dHead], V[T, nKV*dHead] → O[T, nH*dHead]
 * 因果掩码 + 在线 softmax 的批量版本（训练用 T×T 显式矩阵）。
 * 反向为 FlashAttention-2 的标准解析式。
 */
export function causalAttn(Q, K, V, T, nH, nKV, dHead) {
  const scale = 1 / Math.sqrt(dHead);
  const group = Math.max(1, Math.floor(nH / nKV));
  const qD = nH * dHead, kvD = nKV * dHead;
  const out = new Float32Array(T * qD);
  // 缓存每 head 的注意力概率 P[h][t][j]，供反向使用
  const P = new Float32Array(nH * T * T);

  for (let h = 0; h < nH; h++) {
    const kvh = Math.floor(h / group);
    const pb = h * T * T;
    for (let t = 0; t < T; t++) {
      const qb = t * qD + h * dHead;
      // 1) 分数
      let mx = -Infinity;
      const s = new Float32Array(t + 1);
      for (let j = 0; j <= t; j++) {
        const kb = j * kvD + kvh * dHead;
        let acc = 0;
        for (let d = 0; d < dHead; d++) acc += Q.data[qb + d] * K.data[kb + d];
        acc *= scale;
        s[j] = acc;
        if (acc > mx) mx = acc;
      }
      // 2) softmax（因果：只到 t）
      let sum = 0;
      for (let j = 0; j <= t; j++) { s[j] = Math.exp(s[j] - mx); sum += s[j]; }
      const inv = 1 / sum;
      for (let j = 0; j <= t; j++) { s[j] *= inv; P[pb + t * T + j] = s[j]; }
      // 3) 加权求和 V
      const ob = t * qD + h * dHead;
      for (let j = 0; j <= t; j++) {
        const p = s[j];
        if (p === 0) continue;
        const vb = j * kvD + kvh * dHead;
        for (let d = 0; d < dHead; d++) out[ob + d] += p * V.data[vb + d];
      }
    }
  }

  return op(out, (g) => {
    const gQ = (Q.grad ??= new Float32Array(T * qD));
    const gK = (K.grad ??= new Float32Array(T * kvD));
    const gV = (V.grad ??= new Float32Array(T * kvD));
    for (let h = 0; h < nH; h++) {
      const kvh = Math.floor(h / group);
      const pb = h * T * T;
      for (let t = 0; t < T; t++) {
        const ob = t * qD + h * dHead;
        // dP = dO · Vᵀ
        const dP = new Float32Array(t + 1);
        let dotP = 0;
        for (let j = 0; j <= t; j++) {
          const vb = j * kvD + kvh * dHead;
          let acc = 0;
          for (let d = 0; d < dHead; d++) acc += g[ob + d] * V.data[vb + d];
          dP[j] = acc;
          dotP += acc * P[pb + t * T + j];
        }
        const qb = t * qD + h * dHead;
        for (let j = 0; j <= t; j++) {
          const pj = P[pb + t * T + j];
          if (pj === 0) continue;
          const dS = pj * (dP[j] - dotP);
          // dV[j] += P[t,j] * dO[t]
          const vb = j * kvD + kvh * dHead;
          for (let d = 0; d < dHead; d++) gV[vb + d] += pj * g[ob + d];
          // dQ[t] += dS * K[j] * scale
          const kb = j * kvD + kvh * dHead;
          for (let d = 0; d < dHead; d++) {
            gQ[qb + d] += dS * K.data[kb + d] * scale;
            gK[kb + d] += dS * Q.data[qb + d] * scale;
          }
        }
      }
    }
  });
}

// ── SwiGLU（逐行）────────────────────────────────────────────────────
/** O[T,H] = silu(G[T,H]) ⊙ U[T,H] */
export function swigluRows(G, U, T, H) {
  const out = new Float32Array(T * H);
  const sig = new Float32Array(T * H);
  for (let i = 0; i < T * H; i++) {
    const g = G.data[i];
    const s = 1 / (1 + Math.exp(-g));
    sig[i] = s;
    out[i] = g * s * U.data[i];
  }
  return op(out, (g) => {
    const gG = (G.grad ??= new Float32Array(T * H));
    const gU = (U.grad ??= new Float32Array(T * H));
    for (let i = 0; i < T * H; i++) {
      const s = sig[i], gv = G.data[i];
      gG[i] += g[i] * U.data[i] * (s + gv * s * (1 - s));
      gU[i] += g[i] * (gv * s);
    }
  });
}

// ── 批量因果自注意力（块对角掩码）────────────────────────────────────
/**
 * Q[B*T, nH*dHead], K/V[B*T, nKV*dHead] → O[B*T, nH*dHead]
 * 掩码为**块对角因果**：样本 b 的第 t 个 token 只看同一样本的 0..t。
 * 这是训练的标准做法（HF 的 `_attn_mask` 4D 块对角矩阵）。
 * 无 batch 的等价形式由 causalAttn 提供（更快）。
 */
export function causalAttnBatch(Q, K, V, B, T, nH, nKV, dHead) {
  const scale = 1 / Math.sqrt(dHead);
  const group = Math.max(1, Math.floor(nH / nKV));
  const qD = nH * dHead, kvD = nKV * dHead;
  const N = B * T;
  const out = new Float32Array(N * qD);
  const P = new Float32Array(B * nH * T * T);

  for (let b = 0; b < B; b++) {
    const base = b * T;
    for (let h = 0; h < nH; h++) {
      const kvh = Math.floor(h / group);
      const pb = (b * nH + h) * T * T;
      for (let t = 0; t < T; t++) {
        const qb = (base + t) * qD + h * dHead;
        let mx = -Infinity;
        const s = new Float32Array(t + 1);
        for (let j = 0; j <= t; j++) {
          const kb = (base + j) * kvD + kvh * dHead;
          let acc = 0;
          for (let d = 0; d < dHead; d++) acc += Q.data[qb + d] * K.data[kb + d];
          acc *= scale;
          s[j] = acc;
          if (acc > mx) mx = acc;
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) { s[j] = Math.exp(s[j] - mx); sum += s[j]; }
        const inv = 1 / sum;
        for (let j = 0; j <= t; j++) { s[j] *= inv; P[pb + t * T + j] = s[j]; }
        const ob = (base + t) * qD + h * dHead;
        for (let j = 0; j <= t; j++) {
          const p = s[j];
          if (p === 0) continue;
          const vb = (base + j) * kvD + kvh * dHead;
          for (let d = 0; d < dHead; d++) out[ob + d] += p * V.data[vb + d];
        }
      }
    }
  }

  return op(out, (g) => {
    const gQ = (Q.grad ??= new Float32Array(N * qD));
    const gK = (K.grad ??= new Float32Array(N * kvD));
    const gV = (V.grad ??= new Float32Array(N * kvD));
    for (let b = 0; b < B; b++) {
      const base = b * T;
      for (let h = 0; h < nH; h++) {
        const kvh = Math.floor(h / group);
        const pb = (b * nH + h) * T * T;
        for (let t = 0; t < T; t++) {
          const ob = (base + t) * qD + h * dHead;
          const dP = new Float32Array(t + 1);
          let dotP = 0;
          for (let j = 0; j <= t; j++) {
            const vb = (base + j) * kvD + kvh * dHead;
            let acc = 0;
            for (let d = 0; d < dHead; d++) acc += g[ob + d] * V.data[vb + d];
            dP[j] = acc;
            dotP += acc * P[pb + t * T + j];
          }
          const qb = (base + t) * qD + h * dHead;
          for (let j = 0; j <= t; j++) {
            const pj = P[pb + t * T + j];
            if (pj === 0) continue;
            const dS = pj * (dP[j] - dotP);
            const vb = (base + j) * kvD + kvh * dHead;
            const kb = (base + j) * kvD + kvh * dHead;
            for (let d = 0; d < dHead; d++) {
              gV[vb + d] += pj * g[ob + d];
              gQ[qb + d] += dS * K.data[kb + d] * scale;
              gK[kb + d] += dS * Q.data[qb + d] * scale;
            }
          }
        }
      }
    }
  });
}

// ── 线性标量损失（测试与自定义目标用）───────────────────────────────
/** out[0] = Σ_i Y[i]·t[i]   —— 反向把 g[0]·t[i] 传给 Y */
export function weightedSum(Y, t) {
  const n = Y.data.length;
  const out = new Float32Array(1);
  for (let i = 0; i < n; i++) out[0] += Y.data[i] * t[i];
  return op(out, (g) => {
    const gY = (Y.grad ??= new Float32Array(n));
    const k = g[0];
    for (let i = 0; i < n; i++) gY[i] += k * t[i];
  });
}

// ── 交叉熵（均值）────────────────────────────────────────────────────
/** L = -(1/T) Σ_t log P[targets[t]]   —— 反向 dLogits = (P − OneHot)/T */
export function crossEntropyMean(logits, targets, T, V) {
  const out = new Float32Array(1);
  let loss = 0;
  const probs = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    const lb = t * V;
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (logits.data[lb + i] > mx) mx = logits.data[lb + i];
    let s = 0;
    for (let i = 0; i < V; i++) { const e = Math.exp(logits.data[lb + i] - mx); probs[lb + i] = e; s += e; }
    const lse = mx + Math.log(s);
    loss += -(logits.data[lb + targets[t]] - lse);
    for (let i = 0; i < V; i++) probs[lb + i] /= s;
  }
  out[0] = loss / T;
  return op(out, (g) => {
    const gL = (logits.grad ??= new Float32Array(T * V));
    const k = g[0] / T;
    for (let t = 0; t < T; t++) {
      const lb = t * V;
      for (let i = 0; i < V; i++) gL[lb + i] += k * probs[lb + i];
      gL[lb + targets[t]] -= k;
    }
  });
}
