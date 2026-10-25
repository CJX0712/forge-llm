/**
 * verify_export.mjs —— 反向互操作：forge 写出的 GGUF，用 forge 自己的读取引擎
 * 逐张量导出「视图」(dims + 字节 sha256)，供 verify_export.py 与
 * llama.cpp 官方 gguf-py 读出的视图逐项比对。
 *
 * 必要性：forge 的「写→读」往返测试无法发现「写错但读也错」的对称错误
 * （例如 GGUF 字符串长度前缀宽度写错又读错，自洽但不兼容）。
 * 只有拿**第三方实现**（gguf-py）去读 forge 写的文件，才能证伪格式正确性。
 *
 * 用法（在仓库根目录）：node tools/interop/verify_export.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { readGGUF } from "../../src/L9/gguf.js";

const PATH = "tools/interop/out/forge-export.gguf";
const g = readGGUF(new Uint8Array(readFileSync(PATH)));

const view = {};
for (const t of g.tensors) {
  const data = g.getTensor(t.name).data;
  view[t.name] = {
    dims: t.dims,
    type: t.type,
    nbytes: data.length,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

const fields = {};
for (const [k, v] of g.meta) {
  fields[k] = Array.isArray(v) ? { type: "array", len: v.length, sample: v.slice(0, 4) } : v;
}

writeFileSync(
  "tools/interop/out/forge-view.json",
  JSON.stringify({ path: PATH, tensors: view, fields }, null, 2)
);
console.log(`forge 读取引擎视图 → tools/interop/out/forge-view.json（${g.tensors.length} 个张量）`);
