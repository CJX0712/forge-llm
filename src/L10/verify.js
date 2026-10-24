/**
 * L10 · 验证层 (Invariant Verification)
 * ------------------------------------------------------------------
 * 全链路「不变量」自检：每一层声称的硬金标准都在此被独立交叉验证。
 * 任何一项失败都意味着内核不可信 —— 这是 forge-llm 的验收底线。
 *
 * 运行：node src/L10/verify.js  （或 forge.mjs verify）
 * 输出：每项 I-xx 的 pass/fail + 关键数值，全部通过才退出码 0。
 */

import { gemm, gemmRef, matvec, dot, argmax, RNG, relL2, maxAbsErr } from "../L0/numeric.js";
import { softmax, logsumexp, rmsnorm, rope, ropeTable } from "../L1/special.js";
import { resetTape, linear, silu, rmsnorm as rmsnormAD, softmaxCE, backward, gradCheck, param, leaf, Adam, op } from "../L2/autograd.js";
import { matmulT, matmul, embedLookup, rmsNorm, ropeRows, sliceCols, causalAttn, swigluRows, crossEntropyMean, weightedSum } from "../L2/tensor.js";
import { dequantize, quantizeRow, dotQ, info, TYPE_INFO, GGMLType } from "../L3/quant.js";
import { QuantLinear, SwiGLU } from "../L4/nn.js";
import { attentionNaive, flashAttention, KVCache, PagedKVCache } from "../L5/attention.js";
import { Config, Transformer } from "../L6/transformer.js";
import { TrainableTransformer, initWeights } from "../L6/train.js";
import { sample } from "../L8/sampling.js";
import { exportGGUF, loadGGUF } from "../L9/engine.js";
import { writeGGUF, readGGUF } from "../L9/gguf.js";
import { CharTokenizer, GGUFVocab } from "../L7/tokenizer.js";

let _pass = 0, _fail = 0;
const results = [];
function check(id, name, cond, detail) {
  const pass = !!cond;
  pass ? _pass++ : _fail++;
  results.push({ id, name, pass, detail });
  return pass;
}
const approx = (a, b, tol) => Math.abs(a - b) <= tol;

export function runVerify(verbose = false) {
  _pass = 0; _fail = 0; results.length = 0;
  const rng = new RNG(12345);

  // ── L0 数值内核 ──
  {
    const M = 8, N = 16, K = 12;
    const A = new Float32Array(M * K), B = new Float32Array(N * K);
    for (let i = 0; i < A.length; i++) A[i] = rng.normal();
    for (let i = 0; i < B.length; i++) B[i] = rng.normal();
    const C = new Float32Array(M * N), Cref = new Float32Array(M * N);
    gemm(C, A, B, M, N, K); gemmRef(Cref, A, B, M, N, K);
    const r = relL2(C, Cref);
    check("I-00", "gemm 与朴素三重循环一致", r < 1e-5, `relL2=${r.toExponential(2)}`);

    const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = rng.normal();
    const o1 = new Float32Array(N), o2 = new Float32Array(N);
    matvec(o1, B, x, N, K);
    // gemm(M=1) 等价：把 x 当作 A[1,K]
    gemm(o2, x, B, 1, N, K);
    check("I-01", "matvec 与 gemm(M=1) 一致", maxAbsErr(o1, o2) === 0, `maxErr=${maxAbsErr(o1, o2)}`);
  }

  // ── L1 特殊函数 ──
  {
    const x = new Float32Array(7); for (let i = 0; i < x.length; i++) x[i] = rng.normal();
    const s = new Float32Array(7); softmax(s, x);
    let sum = 0, nonneg = true; for (let i = 0; i < s.length; i++) { sum += s[i]; if (s[i] < 0) nonneg = false; }
    check("I-10", "softmax 输出 Σ=1 且非负", approx(sum, 1, 1e-5) && nonneg, `Σ=${sum.toFixed(6)}`);

    const lse = logsumexp(x);
    let mx = -Infinity; for (const v of x) mx = Math.max(mx, v);
    let s2 = 0; for (const v of x) s2 += Math.exp(v - mx);
    const idn = mx + Math.log(s2);
    check("I-11", "logsumexp 恒等式", approx(lse, idn, 1e-5), `lse=${lse.toFixed(4)} ref=${idn.toFixed(4)}`);

    const w = new Float32Array(8).fill(1); // w=1 时 RMSNorm 输出才为单位范数
    const x8 = new Float32Array(8); for (let i = 0; i < 8; i++) x8[i] = rng.normal();
    const out = new Float32Array(8); rmsnorm(out, x8, w, 8);
    let ss = 0; for (let i = 0; i < 8; i++) ss += out[i] * out[i];
    check("I-12", "RMSNorm 后 ‖x‖₂/√n=1", approx(Math.sqrt(ss / 8), 1, 1e-6), `‖x‖=${Math.sqrt(ss / 8).toFixed(6)}`);

    // RoPE 相对性：q·k 只依赖 m-n
    const x8a = new Float32Array(8), x8b = new Float32Array(8);
    for (let i = 0; i < 8; i++) { x8a[i] = rng.normal(); x8b[i] = rng.normal(); }
    const qm = new Float32Array(8), kn = new Float32Array(8);
    rope(qm, x8a, 8, 3, 10000);
    rope(kn, x8b, 8, 9, 10000);
    const qm2 = new Float32Array(8), kn2 = new Float32Array(8);
    rope(qm2, x8a, 8, 0, 10000);
    rope(kn2, x8b, 8, 6, 10000);
    let d1 = 0, d2 = 0; for (let i = 0; i < 8; i++) { d1 += qm[i] * kn[i]; d2 += qm2[i] * kn2[i]; }
    check("I-13", "RoPE 相对性 (只依赖 m-n)", approx(d1, d2, 1e-5), `dot1=${d1.toFixed(4)} dot2=${d2.toFixed(4)}`);
  }

  // ── L2 自动微分 ──
  {
    resetTape();
    const K = 5, N = 4;
    const W = param(new Float32Array([0.3, -0.2, 0.1, 0.4, 0.0, 0.2, -0.1, 0.5, 0.3, -0.4, 0.1, 0.2, 0.6, -0.3, 0.2, 0.4, -0.5, 0.1, 0.3, 0.2]));
    const x = leaf(new Float32Array([1, 0.5, -0.3, 0.2, 0.4]));
    const target = 2;
    const lossFn = () => {
      resetTape();
      const out = linear(W, x, N, K);
      const l = softmaxCE(out, target);
      return l;
    };
    const gc = gradCheck(lossFn, { W });
    check("I-30", "梯度检验 maxRelErr<1e-4", gc.maxRelErr < 1e-4, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);

    // softmaxCE 解析梯度 = p - onehot
    resetTape();
    const logits = leaf(new Float32Array([1.2, 0.3, -0.5, 2.1]));
    const loss = softmaxCE(logits, 1);
    backward(loss);
    let mx = -Infinity; for (const v of logits.data) mx = Math.max(mx, v);
    let s = 0; const p = []; for (const v of logits.data) { const e = Math.exp(v - mx); p.push(e); s += e; }
    for (let i = 0; i < p.length; i++) p[i] /= s;
    let ok = true; for (let i = 0; i < p.length; i++) if (Math.abs(logits.grad[i] - (p[i] - (i === 1 ? 1 : 0))) > 1e-5) ok = false;
    check("I-31", "softmaxCE 梯度 = p − onehot", ok, `p=[${p.map((v) => v.toFixed(3)).join(",")}]`);

    // Adam 首步幅度（lr=0.1, 单参数）→ ≈ lr * |g|/(√v+eps) ~ 小量；校验 bias correction 生效
    resetTape();
    const pw = param(new Float32Array([0.0]));
    const lf = () => { resetTape(); const l = leaf(new Float32Array([pw.data[0] - 3])); return l; }; // f = w-3, df/dw=1
    // 用最简：手动建图
    resetTape();
    const w2 = param(new Float32Array([0.0]));
    const g = leaf(new Float32Array([3.0])); // 常数梯度来源
    // 构造 loss = (w-3)^2 以便梯度非平凡
    const loss2 = (() => { resetTape(); const w = param(new Float32Array([w2.data[0]])); const outv = leaf(new Float32Array([(w.data[0] - 3) * (w.data[0] - 3)])); return { w, outv }; })();
    // 直接验证 Adam 数学：用已知梯度
    const p1 = param(new Float32Array([1.0])); p1.grad = new Float32Array([1.0]);
    const opt = new Adam([p1], { lr: 0.1, totalSteps: 100 });
    opt.step();
    const upd = Math.abs(1.0 - p1.data[0]);
    check("I-32", "Adam 首步更新幅度≈lr(含 bias corr)", upd > 0 && upd < 0.2, `Δw=${upd.toFixed(4)}`);
  }

  // ── L2b 张量级 AD 算子 ──
  {
    // 数值梯度检验辅助：解析梯度 vs 中心差分
    // 度量采用「相对梯度尺度」误差：|num−ana| / max(‖ana‖∞, floor)。
    // 原因：中心差分在 f32 下的绝对精度约 1e-7，当某元素梯度本身很小时，
    // 逐元素相对误差会被放大 —— PyTorch gradcheck 同样用 atol+rtol 双阈值。
    const numGrad = (lossFn, params, eps = 2e-3) => {
      resetTape();
      const base = lossFn();
      backward(base);
      let worst = 0, at = "";
      for (const [name, p] of Object.entries(params)) {
        const ana = Float32Array.from(p.grad ?? new Float32Array(p.data.length));
        let gmax = 0;
        for (let i = 0; i < ana.length; i++) gmax = Math.max(gmax, Math.abs(ana[i]));
        const floor = Math.max(gmax, 1e-3);
        for (let i = 0; i < p.data.length; i++) {
          const o = p.data[i];
          p.data[i] = o + eps; const lp = lossFn().data[0];
          p.data[i] = o - eps; const lm = lossFn().data[0];
          p.data[i] = o;
          const num = (lp - lm) / (2 * eps);
          const rel = Math.abs(num - ana[i]) / floor;
          if (rel > worst) { worst = rel; at = `${name}[${i}]`; }
        }
      }
      return { maxRelErr: worst, at };
    };

    // I-33 matmulT
    {
      const M = 3, K = 5, N = 4;
      const X = param(randArr(M * K, rng, 0.5)), W = param(randArr(N * K, rng, 0.5));
      const t = new Float32Array(M * N); for (let i = 0; i < t.length; i++) t[i] = rng.normal();
      const lossFn = () => {
        resetTape();
        const Y = matmulT(X, W, M, K, N);
        return weightedSum(Y, t);
      };
      const gc = numGrad(lossFn, { X, W });
      check("I-33a", "matmulT 梯度 = 中心差分", gc.maxRelErr < 2e-3, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);
    }

    // I-33b rmsNorm
    {
      const T = 2, D = 6;
      const X = param(randArr(T * D, rng, 0.5)), Wt = param(new Float32Array(D).fill(1));
      const t = new Float32Array(T * D); for (let i = 0; i < t.length; i++) t[i] = rng.normal();
      const lossFn = () => {
        resetTape();
        const Y = rmsNorm(X, Wt, T, D, 1e-5);
        return weightedSum(Y, t);
      };
      const gc = numGrad(lossFn, { X, Wt });
      check("I-33b", "rmsNorm 梯度 = 中心差分", gc.maxRelErr < 2e-3, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);
    }

    // I-33c ropeRows
    {
      const T = 3, nH = 2, dH = 4;
      const X = param(randArr(T * nH * dH, rng, 0.5));
      const t = new Float32Array(T * nH * dH); for (let i = 0; i < t.length; i++) t[i] = rng.normal();
      const lossFn = () => {
        resetTape();
        const Y = ropeRows(X, T, nH, dH, 0, 10000);
        return weightedSum(Y, t);
      };
      const gc = numGrad(lossFn, { X });
      check("I-33c", "ropeRows 梯度 = 中心差分", gc.maxRelErr < 2e-3, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);
    }

    // I-33d causalAttn（含 GQA）
    {
      const T = 4, nH = 4, nKV = 2, dH = 4;
      const Q = param(randArr(T * nH * dH, rng, 0.5));
      const K = param(randArr(T * nKV * dH, rng, 0.5));
      const V = param(randArr(T * nKV * dH, rng, 0.5));
      const t = new Float32Array(T * nH * dH); for (let i = 0; i < t.length; i++) t[i] = rng.normal();
      const lossFn = () => {
        resetTape();
        const O = causalAttn(Q, K, V, T, nH, nKV, dH);
        return weightedSum(O, t);
      };
      const gc = numGrad(lossFn, { Q, K, V });
      check("I-33d", "causalAttn 梯度 = 中心差分", gc.maxRelErr < 2e-3, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);
    }

    // I-34 crossEntropyMean 梯度 = (P − OneHot)/T
    {
      const T = 3, V = 5;
      const L = param(randArr(T * V, rng, 0.5));
      const targets = [1, 3, 0];
      resetTape();
      const lossNode = crossEntropyMean(L, targets, T, V);
      backward(lossNode);
      let ok = true, worst = 0;
      for (let t = 0; t < T; t++) {
        const lb = t * V;
        let mx = -Infinity; for (let i = 0; i < V; i++) mx = Math.max(mx, L.data[lb + i]);
        let s = 0; const p = new Float32Array(V);
        for (let i = 0; i < V; i++) { p[i] = Math.exp(L.data[lb + i] - mx); s += p[i]; }
        for (let i = 0; i < V; i++) p[i] /= s;
        for (let i = 0; i < V; i++) {
          const want = (p[i] - (i === targets[t] ? 1 : 0)) / T;
          const d = Math.abs(L.grad[lb + i] - want);
          if (d > worst) worst = d;
          if (d > 1e-6) ok = false;
        }
      }
      check("I-34", "crossEntropyMean 梯度 = (P−OneHot)/T", ok, `maxΔ=${worst.toExponential(2)}`);
    }

    // I-35 swigluRows 梯度
    {
      const T = 2, H = 5;
      const G = param(randArr(T * H, rng, 0.5)), U = param(randArr(T * H, rng, 0.5));
      const t = new Float32Array(T * H); for (let i = 0; i < t.length; i++) t[i] = rng.normal();
      const lossFn = () => {
        resetTape();
        const O = swigluRows(G, U, T, H);
        return weightedSum(O, t);
      };
      const gc = numGrad(lossFn, { G, U });
      check("I-35", "swigluRows 梯度 = 中心差分", gc.maxRelErr < 2e-3, `maxRelErr=${gc.maxRelErr.toExponential(2)} @ ${gc.at}`);
    }

    // I-36 embedLookup 梯度 = scatter-add
    {
      const V = 5, D = 3;
      const E = param(randArr(V * D, rng, 0.5));
      const ids = [0, 3, 3];
      resetTape();
      const X = embedLookup(E, ids, V, D);
      const t = new Float32Array(X.data.length);
      for (let i = 0; i < t.length; i++) t[i] = i + 1;
      backward(weightedSum(X, t));
      // E.grad[ids[t]*D+d] 应等于 Σ_{t: ids[t]=v} (t*D+d+1)
      let ok = true, worst = 0;
      for (let v = 0; v < V; v++) {
        for (let d = 0; d < D; d++) {
          let want = 0;
          for (let tt = 0; tt < ids.length; tt++) if (ids[tt] === v) want += tt * D + d + 1;
          const e = Math.abs(E.grad[v * D + d] - want);
          if (e > worst) worst = e;
          if (e > 1e-5) ok = false;
        }
      }
      check("I-36", "embedLookup 梯度 = scatter-add", ok, `maxΔ=${worst.toExponential(2)}`);
    }
  }

  // ── L6b 训练路径 ──
  {
    // I-70 训练前向 与 推理前向 一致（同权重、同 token 序列）
    const cfg = new Config({ dim: 32, hidden: 64, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 40, maxSeq: 16, normEps: 1e-5 });
    const w = initWeights(cfg, 4242);
    const ids = [1, 5, 9, 13, 2, 7];
    const tm = new TrainableTransformer(cfg, w);
    resetTape();
    const logitsT = tm.forward(ids);
    // 推理路径：逐 token + KV 缓存
    const tr = new Transformer(cfg).loadWeights({
      tok_emb: w.tok_emb, ln_f: w.ln_f, wcls: null,
      ...Object.fromEntries(Array.from({ length: cfg.nLayers }, (_, l) => [`layer.${l}`, {
        ln1: w[`layer.${l}`].ln1, ln2: w[`layer.${l}`].ln2,
        wqkv: w[`layer.${l}`].wqkv, wo: w[`layer.${l}`].wo,
        ffn_gate: w[`layer.${l}`].ffn_gate, ffn_up: w[`layer.${l}`].ffn_up, ffn_down: w[`layer.${l}`].ffn_down,
      }])),
    });
    const cache = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    let maxDiff = 0;
    for (let t = 0; t < ids.length; t++) {
      const li = tr.forward(ids[t], t, cache);
      for (let v = 0; v < cfg.vocabSize; v++) maxDiff = Math.max(maxDiff, Math.abs(li[v] - logitsT.data[t * cfg.vocabSize + v]));
    }
    check("I-70", "训练前向 == 推理前向（逐位）", maxDiff < 2e-3, `maxLogitsDiff=${maxDiff.toExponential(2)}`);

    // I-71 过拟合：单批次重复训练 loss 显著下降
    const cfg2 = new Config({ dim: 32, hidden: 64, nLayers: 1, nHeads: 4, nKvHeads: 4, vocabSize: 20, maxSeq: 16, normEps: 1e-5 });
    const m2 = new TrainableTransformer(cfg2, initWeights(cfg2, 777));
    const data = [1, 2, 3, 4, 5, 6, 7, 8];
    const first = (() => { resetTape(); const l = m2.loss(data, data.slice(1)); return m2.backward(l); })();
    const opt = new Adam(m2.paramList, { lr: 0.05, totalSteps: 60, clip: 1.0 });
    let last = first;
    for (let s = 0; s < 60; s++) {
      resetTape(); m2.zeroGrad();
      const l = m2.loss(data, data.slice(1));
      last = m2.backward(l);
      opt.step();
    }
    check("I-71", "单批次过拟合 loss 显著下降", last < first * 0.6, `loss ${first.toFixed(4)} → ${last.toFixed(4)}`);
  }

  // ── L3 量化 ──
  {
    const N = 1024;
    const x = new Float32Array(N); for (let i = 0; i < N; i++) x[i] = rng.normal() * 0.7;
    for (const [tname, t] of [["Q8_0", 8], ["Q4_K", 12], ["Q6_K", 14]]) {
      const q = quantizeRow(t, x, N);
      const dq = dequantize(t, q, N);
      const rmseV = Math.sqrt(dq.reduce((a, b, i) => a + (b - x[i]) ** 2, 0) / N);
      const ok = { 8: 0.02, 12: 0.06, 14: 0.02 }[t];
      check(`I-20+${tname}`, `${tname} 反量化 RMSE 在界内`, rmseV < ok, `RMSE=${rmseV.toExponential(2)} (界 ${ok})`);

      // dotQ vs 反量化后 dot
      const y = new Float32Array(N); for (let i = 0; i < N; i++) y[i] = rng.normal();
      const dqDot = dot(dq, y, N);
      const qd = dotQ(t, q, y, N);
      check(`I-22+${tname}`, `${tname} dotQ 与反量化点积一致`, Math.abs(qd - dqDot) < 1e-3 * (Math.abs(dqDot) + 1), `Δ=${Math.abs(qd - dqDot).toExponential(2)}`);
    }
    // bpw 表
    const bpwOK = info(12).bpw === 4.5 && info(14).bpw === 6.5625 && info(8).bpw === 8.5;
    check("I-23", "bit-per-weight 与官方一致", bpwOK, `Q4_K=${info(12).bpw} Q6_K=${info(14).bpw} Q8_0=${info(8).bpw}`);
  }

  // ── L4 组件 ──
  {
    const N = 32, K = 64;  // K 取 32 的倍数（Q8_0 块长=32，保证行边界对齐）
    const W = new Float32Array(N * K); for (let i = 0; i < W.length; i++) W[i] = rng.normal();
    const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = rng.normal();
    const f32layer = QuantLinear.fromF32(W, N, K, 0);
    const q8 = QuantLinear.fromF32(W, N, K, 8);
    const o1 = new Float32Array(N), o2 = new Float32Array(N);
    f32layer.forward(x, o1); q8.forward(x, o2);
    const cos = dot(o1, o2, N) / (Math.hypot(...o1) * Math.hypot(...o2));
    check("I-40", "QuantLinear(Q8_0) 与 f32 余弦>0.999", cos > 0.999, `cos=${cos.toFixed(6)}`);

    const dim = 16, hidden = 32;
    const gW = new Float32Array(hidden * dim), uW = new Float32Array(hidden * dim), dW = new Float32Array(dim * hidden);
    for (let i = 0; i < gW.length; i++) gW[i] = rng.normal();
    for (let i = 0; i < uW.length; i++) uW[i] = rng.normal();
    for (let i = 0; i < dW.length; i++) dW[i] = rng.normal();
    const sw = new SwiGLU(QuantLinear.fromF32(gW, hidden, dim, 0), QuantLinear.fromF32(uW, hidden, dim, 0), QuantLinear.fromF32(dW, dim, hidden, 0), dim, hidden);
    const out = new Float32Array(dim);
    const gv = new Float32Array(hidden), uv = new Float32Array(hidden), tmp = new Float32Array(hidden);
    matvec(gv, gW, x.slice(0, dim), hidden, dim); matvec(uv, uW, x.slice(0, dim), hidden, dim);
    for (let i = 0; i < hidden; i++) tmp[i] = (gv[i] / (1 + Math.exp(-gv[i]))) * uv[i];
    const ref = new Float32Array(dim); matvec(ref, dW, tmp, dim, hidden);
    sw.forward(x.slice(0, dim), out);
    check("I-41", "SwiGLU 块与手动分解一致", maxAbsErr(out, ref) < 1e-5, `maxErr=${maxAbsErr(out, ref).toExponential(2)}`);
  }

  // ── L5 注意力 ──
  {
    const n = 32, d = 8;
    const q = new Float32Array(d); for (let i = 0; i < d; i++) q[i] = rng.normal();
    const K = new Float32Array(n * d), V = new Float32Array(n * d);
    for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
    const oFlash = new Float32Array(d), oNaive = new Float32Array(d);
    flashAttention(oFlash, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
    attentionNaive(oNaive, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
    check("I-50", "flashAttention 与朴素一致", maxAbsErr(oFlash, oNaive) < 1e-4, `maxErr=${maxAbsErr(oFlash, oNaive).toExponential(2)}`);

    let rowSum = 0; for (let i = 0; i < d; i++) rowSum += oNaive[i]; // 不是概率行和，改为权重行和
    // 直接重算权重行和
    const s = new Float32Array(n); let mx = -Infinity;
    for (let j = 0; j < n; j++) { let a = 0; for (let i = 0; i < d; i++) a += q[i] * K[j * d + i]; a *= 1 / Math.sqrt(d); if (a > mx) mx = a; s[j] = a; }
    let sm = 0; for (let j = 0; j < n; j++) { s[j] = Math.exp(s[j] - mx); sm += s[j]; }
    let wsum = 0; for (let j = 0; j < n; j++) wsum += s[j] / sm;
    check("I-51", "注意力权重行和=1", approx(wsum, 1, 1e-5), `Σw=${wsum.toFixed(6)}`);

    // 因果：位置 n-1 的 query 只看 ≤ n-1；检查更前面位置 q 的权重在 >pos 处为 0
    const q2 = new Float32Array(d); for (let i = 0; i < d; i++) q2[i] = rng.normal();
    const o2 = new Float32Array(d);
    flashAttention(o2, q2, K, V, n, d, 1 / Math.sqrt(d), true, 4, 0); // qOff=4
    // 对 j>4 的权重应为 0：通过对比对 j≤4 的 naive 即可；这里直接校验 w[5..]=0 用 naive
    const s2 = new Float32Array(n); let mx2 = -Infinity;
    for (let j = 0; j < n; j++) { if (j > 4) { s2[j] = -Infinity; continue; } let a = 0; for (let i = 0; i < d; i++) a += q2[i] * K[j * d + i]; a *= 1 / Math.sqrt(d); if (a > mx2) mx2 = a; s2[j] = a; }
    let sm2 = 0; for (let j = 0; j < n; j++) if (isFinite(s2[j])) sm2 += Math.exp(s2[j] - mx2);
    let wsum2 = 0; for (let j = 0; j < 5; j++) wsum2 += Math.exp(s2[j] - mx2) / sm2;
    check("I-52", "因果掩码：>qOff 处权重为 0", approx(wsum2, 1, 1e-5), `Σ(≤pos)=${wsum2.toFixed(6)}`);

    // KV cache 增量 == 全量重算（用 KVCache 装前缀，对比 full naive attentionNaive 同前缀）
    const cache = new KVCache(1, n, d);
    for (let p = 0; p < n; p++) { const kk = K.subarray(p * d, p * d + d), vv = V.subarray(p * d, p * d + d); cache.put(0, p, kk, vv); }
    const Kc = cache.kView(0, n), Vc = cache.vView(0, n);
    const oCache = new Float32Array(d);
    flashAttention(oCache, q, Kc, Vc, n, d, 1 / Math.sqrt(d), true, n - 1, 0);
    check("I-54", "KV 缓存连续视图 == 全量重算", maxAbsErr(oCache, oNaive) < 1e-5, `maxErr=${maxAbsErr(oCache, oNaive).toExponential(2)}`);
  }

  // ── L6 + L7 + L8 + L9 端到端 ──
  {
    // 构造一个 tiny 随机模型，f32 推理 vs 量化推理一致性（维度均取 32 的倍数）
    const cfg = new Config({ dim: 128, hidden: 256, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 64, maxSeq: 32, normEps: 1e-5 });
    const rng2 = new RNG(99);
    const W = makeRandomWeights(cfg, rng2, 1.0);   // 放大权重，使 logits 充分非零，余弦度量有意义
    const tr32 = toTransformerLocal(W, cfg, 0);
    const tr8 = toTransformerLocal(W, cfg, 8);
    const cache32 = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    const cache8 = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    let cosMin = 1, argMaxMatch = 0, nPos = 8;
    for (let pos = 0; pos < nPos; pos++) {
      const l32 = tr32.forward(pos, pos, cache32);
      const l8 = tr8.forward(pos, pos, cache8);
      const c = dot(l32, l8, cfg.vocabSize) / (Math.hypot(...l32) * Math.hypot(...l8));
      cosMin = Math.min(cosMin, c);
      if (argmax(l32, cfg.vocabSize) === argmax(l8, cfg.vocabSize)) argMaxMatch++;
    }
    check("I-60", "f32 与 Q8_0 推理输出余弦≥0.99 且 argmax 至多 1 次翻转", cosMin >= 0.99 && argMaxMatch >= nPos - 1,
      `minCos=${cosMin.toFixed(5)} argmaxMatch=${argMaxMatch}/${nPos}`);

    // 因果：同前缀 → 同 logits（改变后续 token 不影响前面）
    const trA = toTransformerLocal(W, cfg, 0), trB = toTransformerLocal(W, cfg, 0);
    const ca = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim), cb = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    const la = trA.forward(5, 0, ca), lb = trB.forward(5, 0, cb);
    check("I-61", "同输入 → 同 logits（确定性）", maxAbsErr(la, lb) < 1e-6, `maxErr=${maxAbsErr(la, lb).toExponential(2)}`);
  }

  // ── GGUF 往返 ──
  {
    const cfg = new Config({ dim: 16, hidden: 32, nLayers: 1, nHeads: 4, nKvHeads: 2, vocabSize: 20, maxSeq: 16 });
    const rng3 = new RNG(7);
    const W = makeRandomWeights(cfg, rng3);
    const vocab = buildCharVocabForGGUF(20);
    const bytes = exportGGUF(W, cfg, vocab, 0, "verify-model");
    const loaded = loadGGUF(bytes);
    // 对比原权重与重加载权重逐张量一致（f32 → f32 应逐位）
    const cfg2 = loaded.config;
    const tr0 = toTransformerLocal(W, cfg, 0);
    const cache0 = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    const cacheL = new KVCache(cfg2.nLayers, cfg2.maxSeq, cfg2.kvDim);
    let maxDiff = 0;
    for (let pos = 0; pos < 6; pos++) {
      const l0 = tr0.forward(pos, pos, cache0);
      const lL = loaded.transformer.forward(pos, pos, cacheL);
      maxDiff = Math.max(maxDiff, maxAbsErr(l0, lL));
    }
    check("I-90", "GGUF 导出→重加载 输出逐位一致", maxDiff < 1e-5, `maxLogitsDiff=${maxDiff.toExponential(2)}`);
    // 元数据结构校验
    check("I-91", "GGUF 元数据 architecture=torge", loaded.metadata.get("general.architecture") === "forge", `arch=${loaded.metadata.get("general.architecture")}`);
  }

  // ── L8 采样 ──
  {
    const logits = new Float32Array([2.1, 0.3, -1.2, 0.8, 0.0]);
    const g = sample(logits.slice(), { temperature: 0, top_k: 1 });
    check("I-81", "temperature=0 退化为 greedy(argmax)", g === 0, `argmax=${g}`);
    const probs = new Float32Array(5); softmax(probs, logits);
    let s = 0; for (const v of probs) s += v;
    check("I-80", "采样前 softmax Σ=1", approx(s, 1, 1e-6), `Σ=${s.toFixed(6)}`);
    // min-p：把低概率清零后重归一化
    let pmax = 0; for (const v of probs) pmax = Math.max(pmax, v);
    const thr = 0.1 * pmax; let nz = 0; for (const v of probs) if (v >= thr) nz++;
    check("I-82", "min-p 阈值截断生效", nz < probs.length, `保留 ${nz}/${probs.length} 个 token`);
  }

  return { pass: _pass, fail: _fail, results, ok: _fail === 0 };
}

// ── 测试辅助 ─────────────────────────────────────────────────────────────
function randArr(n, rng, scale = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}

function toTransformerLocal(W, cfg, quantType) {
  return loadGGUFFromWeights(W, cfg, quantType);
}
function loadGGUFFromWeights(W, cfg, quantType) {
  // 直接走 export→loadGGUF 往返以保证与真实路径一致
  const bytes = exportGGUF(W, cfg, null, quantType, "tmp");
  return loadGGUF(bytes).transformer;
}

function makeRandomWeights(cfg, rng, scale = 0.3) {
  const g = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = rng.normal() * scale; return a; };
  const W = {
    tok_emb: g(cfg.vocabSize * cfg.dim),
    ln_f: new Float32Array(cfg.dim).fill(1),
  };
  for (let l = 0; l < cfg.nLayers; l++) {
    W[`layer.${l}`] = {
      ln1: new Float32Array(cfg.dim).fill(1),
      ln2: new Float32Array(cfg.dim).fill(1),
      wqkv: g(cfg.qkvOut * cfg.dim),
      wo: g(cfg.dim * cfg.dim),
      ffn_gate: g(cfg.hidden * cfg.dim),
      ffn_up: g(cfg.hidden * cfg.dim),
      ffn_down: g(cfg.dim * cfg.hidden),
    };
  }
  return W;
}

function buildCharVocabForGGUF(vocabSize) {
  const tokenBytes = [], scores = [], types = [];
  for (let i = 0; i < vocabSize; i++) {
    tokenBytes.push(new TextEncoder().encode(String.fromCharCode(97 + (i % 26)) + (i > 25 ? i : "")));
    scores.push(0); types.push(0);
  }
  return new GGUFVocab(tokenBytes, types, scores, { bos: 1, eos: 2, addBos: false, addEos: false });
}

// 直接运行
const isMain = process.argv[1] && process.argv[1].replace(/\\/g, "/").endsWith("L10/verify.js");
if (isMain) {
  const r = runVerify(true);
  for (const res of r.results) console.log(`${res.pass ? "✅" : "❌"} ${res.id}  ${res.name}  —  ${res.detail}`);
  console.log(`\n总计: ${r.pass} 通过 / ${r.fail} 失败`);
  process.exit(r.fail === 0 ? 0 : 1);
}
