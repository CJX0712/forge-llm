/**
 * 单元测试 · L2 自动微分 & L2b 张量算子 & L6 训练
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  resetTape, linear, silu, rmsnorm, softmaxCE, backward, gradCheck, param, leaf, Adam, op,
} from "../src/L2/autograd.js";
import {
  matmulT, embedLookup, rmsNorm, ropeRows, causalAttn, causalAttnBatch,
  swigluRows, crossEntropyMean, weightedSum,
} from "../src/L2/tensor.js";
import { Config } from "../src/L6/transformer.js";
import { TrainableTransformer, initWeights } from "../src/L6/train.js";
import { RNG } from "../src/L0/numeric.js";

/** 中心差分：返回某参数的最大相对梯度误差（相对梯度尺度） */
function numGradErr(lossFn, params, eps = 2e-3) {
  resetTape();
  const base = lossFn();
  backward(base);
  let worst = 0, at = "";
  for (const [name, p] of Object.entries(params)) {
    const ana = Float32Array.from(p.grad ?? new Float32Array(p.data.length));
    let gmax = 0;
    for (const v of ana) gmax = Math.max(gmax, Math.abs(v));
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
  return { worst, at };
}

test("linear 梯度检验", () => {
  const rng = new RNG(1);
  const K = 6, N = 5;
  const W = param(new Float32Array(K * N).map(() => rng.normal()));
  const x = leaf(new Float32Array(K).map(() => rng.normal()));
  const t = new Float32Array(N).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    const Y = linear(W, x, N, K);
    return weightedSum(Y, t);
  };
  const { worst } = numGradErr(lossFn, { W });
  assert.ok(worst < 2e-3, `worst=${worst}`);
});

test("softmaxCE 梯度 = p − onehot", () => {
  resetTape();
  const logits = leaf(new Float32Array([1.2, 0.3, -0.5, 2.1]));
  backward(softmaxCE(logits, 1));
  let mx = -Infinity; for (const v of logits.data) mx = Math.max(mx, v);
  let s = 0; const p = [];
  for (const v of logits.data) { const e = Math.exp(v - mx); p.push(e); s += e; }
  for (let i = 0; i < p.length; i++) {
    assert.ok(Math.abs(logits.grad[i] - (p[i] / s - (i === 1 ? 1 : 0))) < 1e-5);
  }
});

test("matmulT 梯度检验", () => {
  const rng = new RNG(2);
  const M = 4, K = 5, N = 3;
  const X = param(new Float32Array(M * K).map(() => rng.normal()));
  const W = param(new Float32Array(N * K).map(() => rng.normal()));
  const t = new Float32Array(M * N).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    return weightedSum(matmulT(X, W, M, K, N), t);
  };
  const { worst } = numGradErr(lossFn, { X, W });
  assert.ok(worst < 2e-3, `worst=${worst}`);
});

test("rmsNorm 梯度检验", () => {
  const rng = new RNG(3);
  const T = 3, D = 8;
  const X = param(new Float32Array(T * D).map(() => rng.normal()));
  const W = param(new Float32Array(D).fill(1));
  const t = new Float32Array(T * D).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    return weightedSum(rmsNorm(X, W, T, D, 1e-5), t);
  };
  const { worst } = numGradErr(lossFn, { X, W });
  assert.ok(worst < 2e-3, `worst=${worst}`);
});

test("ropeRows 梯度检验（正交旋转的转置）", () => {
  const rng = new RNG(4);
  const T = 3, nH = 2, dH = 4;
  const X = param(new Float32Array(T * nH * dH).map(() => rng.normal()));
  const t = new Float32Array(T * nH * dH).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    return weightedSum(ropeRows(X, T, nH, dH, 0, 10000), t);
  };
  const { worst } = numGradErr(lossFn, { X });
  assert.ok(worst < 2e-3, `worst=${worst}`);
});

test("causalAttn 梯度检验", () => {
  const rng = new RNG(5);
  const T = 4, nH = 4, nKV = 2, dH = 4;
  const Q = param(new Float32Array(T * nH * dH).map(() => rng.normal()));
  const K = param(new Float32Array(T * nKV * dH).map(() => rng.normal()));
  const V = param(new Float32Array(T * nKV * dH).map(() => rng.normal()));
  const t = new Float32Array(T * nH * dH).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    return weightedSum(causalAttn(Q, K, V, T, nH, nKV, dH), t);
  };
  const { worst } = numGradErr(lossFn, { Q, K, V });
  assert.ok(worst < 5e-3, `worst=${worst}`);
});

test("swigluRows 梯度检验", () => {
  const rng = new RNG(6);
  const T = 2, H = 5;
  const G = param(new Float32Array(T * H).map(() => rng.normal()));
  const U = param(new Float32Array(T * H).map(() => rng.normal()));
  const t = new Float32Array(T * H).map(() => rng.normal());
  const lossFn = () => {
    resetTape();
    return weightedSum(swigluRows(G, U, T, H), t);
  };
  const { worst } = numGradErr(lossFn, { G, U });
  assert.ok(worst < 2e-3, `worst=${worst}`);
});

test("crossEntropyMean 梯度 = (P−OneHot)/T", () => {
  resetTape();
  const rng = new RNG(7);
  const T = 3, V = 6;
  const L = param(new Float32Array(T * V).map(() => rng.normal()));
  const targets = [1, 3, 0];
  backward(crossEntropyMean(L, targets, T, V));
  for (let t = 0; t < T; t++) {
    const lb = t * V;
    let mx = -Infinity; for (let i = 0; i < V; i++) mx = Math.max(mx, L.data[lb + i]);
    let s = 0; const p = new Float32Array(V);
    for (let i = 0; i < V; i++) { p[i] = Math.exp(L.data[lb + i] - mx); s += p[i]; }
    for (let i = 0; i < V; i++) {
      const want = (p[i] / s - (i === targets[t] ? 1 : 0)) / T;
      assert.ok(Math.abs(L.grad[lb + i] - want) < 1e-5);
    }
  }
});

test("embedLookup 梯度 = scatter-add", () => {
  resetTape();
  const V = 4, D = 3;
  const E = param(new Float32Array(V * D).map((_, i) => i * 0.1));
  const ids = [0, 2, 2];
  const X = embedLookup(E, ids, V, D);
  const t = new Float32Array(X.data.length).map((_, i) => i + 1);
  backward(weightedSum(X, t));
  for (let v = 0; v < V; v++) {
    for (let d = 0; d < D; d++) {
      let want = 0;
      for (let tt = 0; tt < ids.length; tt++) if (ids[tt] === v) want += tt * D + d + 1;
      assert.ok(Math.abs(E.grad[v * D + d] - want) < 1e-5);
    }
  }
});

test("批量块对角掩码 == 逐样本独立计算", () => {
  const cfg = new Config({ dim: 32, hidden: 64, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 20, maxSeq: 32 });
  const m = new TrainableTransformer(cfg, initWeights(cfg, 11));
  const rows = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]];
  resetTape();
  const batchLoss = m.backward(m.lossBatch(rows));
  let sum = 0;
  for (const r of rows) {
    resetTape();
    sum += m.backward(m.loss(r.slice(0, -1), r.slice(1)));
  }
  assert.ok(Math.abs(batchLoss - sum / rows.length) < 1e-5, `${batchLoss} vs ${sum / rows.length}`);
});

test("单批次过拟合：loss 显著下降", () => {
  const cfg = new Config({ dim: 32, hidden: 64, nLayers: 1, nHeads: 4, nKvHeads: 4, vocabSize: 20, maxSeq: 16 });
  const m = new TrainableTransformer(cfg, initWeights(cfg, 13));
  const data = [1, 2, 3, 4, 5, 6, 7, 8];
  resetTape();
  const first = m.backward(m.loss(data, data.slice(1)));
  const opt = new Adam(m.paramList, { lr: 0.05, totalSteps: 60 });
  let last = first;
  for (let i = 0; i < 60; i++) {
    resetTape(); m.zeroGrad();
    last = m.backward(m.loss(data, data.slice(1)));
    opt.step();
  }
  assert.ok(last < first * 0.6, `${first} → ${last}`);
});

test("Adam 含 bias correction：首步更新 ≈ lr", () => {
  const p = param(new Float32Array([1.0]));
  p.grad = new Float32Array([1.0]);
  const opt = new Adam([p], { lr: 0.1, totalSteps: 100 });
  opt.step();
  assert.ok(Math.abs(1.0 - p.data[0] - 0.1) < 1e-4);
});

test("tape 隔离：resetTape 后旧节点不参与反向", () => {
  resetTape();
  const p = param(new Float32Array([1.0]));
  const l1 = op(new Float32Array([p.data[0]]), () => {});
  resetTape();
  const l2 = op(new Float32Array([p.data[0] * 2]), () => {});
  backward(l2);
  assert.ok(l2.grad);
});
