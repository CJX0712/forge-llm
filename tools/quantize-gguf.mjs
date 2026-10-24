/**
 * tools · GGUF 量化器
 * ------------------------------------------------------------------
 * 把 f32/F16 的 GGUF 权重按行量化为 Q4_K / Q5_K / Q6_K / Q8_0 / F16，
 * 输出仍是**标准 GGUF v3**，可直接被 llama.cpp / llama-cli / ollama 加载。
 *
 * 复用 llama.cpp 的成熟做法：
 *  - 按行（row）量化：GGUF 的每个量化张量以「行」为单位存储，
 *    K 维长度必须是块长的整数倍（Q4_K/Q6_K=256，Q8_0=32）。
 *  - 归一化向量层（attn_norm / ffn_norm / output_norm）保持 f32 不量化，
 *    与 llama.cpp 的 `--allow-requantize` 默认行为一致。
 *  - 输出头（output.weight）与 token embedding 保持 f32：llama.cpp 亦如此
 *    （量化 embedding 收益小、风险大）。
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { readGGUF, writeGGUF } from "../src/L9/gguf.js";
import { quantizeRow, dequantize, info, rowBytes } from "../src/L3/quant.js";

/** 需要量化的大矩阵张量名模式（llama 风格命名） */
function shouldQuantize(name, dims) {
  if (dims.length !== 2) return false;              // 一维归一化向量不量化
  if (name.endsWith(".norm.weight")) return false;  // attn_norm / ffn_norm
  if (name === "output_norm.weight") return false;
  if (name === "output.weight") return false;        // 输出头保持 f32
  if (name === "token_embd.weight") return false;    // embedding 保持 f32
  return true;
}

/**
 * @param inPath   输入 GGUF（f32）
 * @param outPath  输出 GGUF（量化）
 * @param opts     {type: 12|13|14|8|1}
 */
export async function quantizeGGUF(inPath, outPath, opts = {}) {
  const type = opts.type ?? 12;
  if (!existsSync(inPath)) throw new Error(`输入文件不存在: ${inPath}`);
  const ti = info(type);

  const buf = new Uint8Array(readFileSync(inPath));
  const g = readGGUF(buf);

  const outTensors = [];
  let origBytes = 0, newBytes = 0, nQuant = 0;

  for (const t of g.tensors) {
    const nElem = t.dims.reduce((a, b) => a * b, 1);
    const src = g.getTensor(t.name);
    if (!src) throw new Error(`读取张量失败: ${t.name}`);
    origBytes += src.data.byteLength;

    if (shouldQuantize(t.name, t.dims) && src.type === 0) {
      const K = t.dims[1];  // 行长度
      const need = ti.blockSize;
      if (K % need !== 0) {
        // 行长不满足块对齐 → 该张量保持 f32
        outTensors.push({ name: t.name, dims: t.dims, type: 0, data: src.data });
        newBytes += src.data.byteLength;
        continue;
      }
      const f32 = new Float32Array(src.data.buffer, src.data.byteOffset, nElem);
      const q = quantizeRow(type, f32, nElem);
      outTensors.push({ name: t.name, dims: t.dims, type, data: q });
      newBytes += q.byteLength;
      nQuant++;
    } else {
      outTensors.push({ name: t.name, dims: t.dims, type: src.type, data: src.data });
      newBytes += src.data.byteLength;
    }
  }

  const meta = new Map(g.meta);
  meta.set("general.quantization_version", 2);
  meta.set("forge.quant_type", ti.name);

  const { bytes } = writeGGUF({ meta, tensors: outTensors });
  const dir = dirname(outPath);
  if (dir && !existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(outPath, bytes);

  return {
    inPath, outPath, type: ti.name, bpw: ti.bpw,
    quantizedTensors: nQuant,
    origBytes, newBytes,
    ratio: origBytes / newBytes,
    fileBytes: bytes.length,
  };
}

// CLI 入口
if (process.argv[1] && process.argv[1].replace(/\\/g, "/").endsWith("tools/quantize-gguf.mjs")) {
  const inPath = process.argv[2], outPath = process.argv[3];
  const typeIdx = process.argv.indexOf("--type");
  const type = typeIdx > 0 ? Number(process.argv[typeIdx + 1]) : 12;
  if (!inPath || !outPath) {
    console.error("用法: node tools/quantize-gguf.mjs <in.gguf> <out.gguf> [--type 12]");
    process.exit(1);
  }
  const r = await quantizeGGUF(inPath, outPath, { type });
  console.log(`✓ ${r.inPath} → ${r.outPath}`);
  console.log(`  类型 ${r.type} (${r.bpw} bpw) · 量化张量 ${r.quantizedTensors} 个`);
  console.log(`  权重 ${(r.origBytes / 1048576).toFixed(2)} MB → ${(r.newBytes / 1048576).toFixed(2)} MB (${r.ratio.toFixed(2)}×)`);
}
