#!/usr/bin/env node
// organoid-mcp — WorldLedger 质检内核的本地 stdio MCP 薄壳。
// 所有计算发生在 ORGANOID_API 指向的服务端；本进程只做协议翻译与文件收发。
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const API = process.env.ORGANOID_API ?? "http://127.0.0.1:8663";

const VALIDATORS = "quality, pairing, kinematics, physics, motion_language, sensors, source_media, hand_video, visual, task_scene";
const POLICIES = "free_root_biped_v1, fixed_base_manipulator_v1, mobile_manipulator_v1, umi_source_v1, source_quality_v1, visual_only_v1, simulation_task_v1";

const GUIDE = `organoid MCP 使用指南（数据质检工作流）

这是什么：organoid_kernel 是具身机器人数据集质检内核。9 种格式自动探测（LeRobot v2.1/v3.0、rosbag、
通用/傅利叶 HDF5、UMI-Zarr、GenDAS、纯视频 MP4、Pico VR、legacy organoid）→ 无损读入统一证据包 →
按数据实际具备的模态跑 10 个验证器 → 输出逐项裁决账本 + 人读报告。

标准工作流：
1. organoid_inspect(path) —— 先看数据长什么样：识别出的格式、有哪些流、能力清单。秒级。
2. organoid_run(path) —— 单条全链验证 → job_id → organoid_result 轮询。约 1-5s；含 hand_video 时 5-30s。
3. organoid_batch(path) —— 整目录批量（默认 skip visual）→ batch-summary 汇总。耗时≈单条×条数。
4. organoid_result(job_id) —— 拿 grade、逐 claim 裁决、quality-report.md 全文、产物清单。
5. organoid_artifact(job_id, path, save_to) —— 把 receipt/keyframe/视频等产物拉回本地细查。
6. organoid_golden_compare() —— 回归对账：新账本 vs golden 冻结基线（默认 runs_leju vs golden/leju_vendor，
   44 条，差异逐条列出）。
7. organoid_experiment(name) —— 两个固定演示实验：grasp_transplant（抓取移植 8 变体，~40s）、
   fullbody_replay（全身动力学重放+抗推，~50s）。数据源硬编码为服务器上的乐聚镜像。

读结果必须懂的两套语义：
- claim 六态：accepted / rejected / inconclusive / not_evaluated / not_applicable / error。
  ⚠ not_evaluated 意为"证据不足以评判"（如缺少必需输入通道），不是"不合格"；
  not_applicable 是"此数据形态天然不适用"。二者都不该被当成缺陷报告给用户。
- 最终六档：accepted / accepted_repaired / accepted_with_warnings / rejected / not_evaluable / error。
  accepted_repaired 表示预检判负后经定向修复+物理重验通过——修了什么看 ledger.repairs。

rejected/not_evaluable 排查路径：grade_reason → 对应 claim 的 reason → 用 organoid_artifact 拉
receipt-<validator>.json 看数值细节。

路径语义：path 参数是【服务端可访问的路径】。相对路径相对内核根——内置样例 "samples/WL_01_04(整理货架)"
开箱即用；乐聚镜像在 ~/leju_vendor/raw。小数据集（≤512MB）可用 organoid_upload 传上去换回服务器路径；
大数据集请自行 rsync 到服务器。

参数合法值：validator（--skip 用）：${VALIDATORS}；policy：${POLICIES}；profile：biped_s200049（当前唯一机型档案）。
服务器已配 Blender 5.2 与 mediapipe：visual 与 hand_video 验证器均可用（organoid_batch 默认 skip visual 是为省时，不是能力缺失）。机器人数据"画面未检出人手"是正确行为，不是缺陷。`;

const server = new McpServer(
  { name: "organoid", version: "0.1.1" },
  { instructions: "organoid 验证具身机器人数据集（LeRobot/rosbag/HDF5/Zarr/VR 等 9 格式）：格式探测、运动学/物理/传感器/视频 10 项验证、六态裁决账本。首次使用先调 organoid_guide 工具读工作流程与结果语义。关键：claim 状态 not_evaluated 意为'证据不足'而非'不合格'；path 参数是服务端可访问的路径，不是本地路径。" }
);

const T = (x) => ({ content: [{ type: "text", text: typeof x === "string" ? x : JSON.stringify(x, null, 1) }] });
const ERR = (e) => ({ isError: true, content: [{ type: "text", text: `${e.message ?? e}\n(organoid 服务地址: ${API} —— 不通时检查the service process and its configured API address)` }] });

async function api(method, p, body) {
  const r = await fetch(`${API}${p}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${p} -> HTTP ${r.status}: ${text.slice(0, 500)}`);
  try { return JSON.parse(text); } catch { return text; }
}

server.tool(
  "organoid_guide",
  "首次使用先调我：返回 organoid 质检工作流指南——工具调用顺序、claim 六态与最终六档的准确语义（not_evaluated≠不合格）、rejected 排查路径、path 参数语义、policy/profile/skip 合法值。本工具不访问网络。",
  {},
  async () => T(GUIDE)
);

server.tool(
  "organoid_inspect",
  "看一个数据集长什么样（不做验证）：自动探测格式，返回完整证据包 summary——有哪些流、各流形状/时长、provenance。path 是服务端可访问的路径（相对路径相对内核根，内置样例 samples/WL_01_04(整理货架)）。秒级同步返回。",
  { path: z.string().describe("服务器上的数据集目录或文件路径"), episode: z.number().int().default(0) },
  async ({ path: p, episode }) => { try { return T(await api("POST", "/inspect", { path: p, episode })); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_run",
  `对单条 episode 跑全链验证（inventory→身份核定→策略→10 验证器→修复回路→账本+报告），返回 job_id，用 organoid_result 轮询。约 1-5s，含 hand_video 时 5-30s。可选：policy(${POLICIES})、profile(biped_s200049)、skip(${VALIDATORS})。`,
  {
    path: z.string().describe("服务器上的数据集路径"),
    episode: z.number().int().default(0),
    policy: z.string().optional(),
    profile: z.string().optional(),
    skip: z.array(z.string()).default([]),
  },
  async (a) => { try { return T(await api("POST", "/run", a)); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_batch",
  "对整个数据集目录批量验证（episode 数自动推断），返回 job_id。完成后 result 里有 batch-summary（逐条 grade + 分布统计）。默认 skip=[\"visual\"]；hand_video 是主要耗时项（视频长的每条 5-30s），赶时间可加进 skip。",
  {
    path: z.string().describe("服务器上的数据集目录"),
    limit: z.number().int().optional().describe("只跑前 N 条"),
    policy: z.string().optional(),
    profile: z.string().optional(),
    skip: z.array(z.string()).default(["visual"]),
  },
  async (a) => { try { return T(await api("POST", "/batch", a)); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_experiment",
  "跑固定演示实验（数据源硬编码为服务器 ~/leju_vendor 乐聚镜像，全局串行）：grasp_transplant=抓取动作模组移植到 8 个换物变体的可信混合重放（~40s，产出力学曲线图+对比视频）；fullbody_replay=整机全动力学重放 49s 轨迹（额定限矩，基线不站稳时侧推对照由协议闸门控）（~50s，产出视频）。返回 job_id。",
  { name: z.enum(["grasp_transplant", "fullbody_replay"]) },
  async (a) => { try { return T(await api("POST", "/experiment", a)); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_result",
  "轮询作业结果。done 后返回：grade（六档）、grade_reason、逐 claim 裁决（六态，注意 not_evaluated=证据不足≠不合格）、quality-report.md 全文、产物清单（用 organoid_artifact 下载）。failed 时看 stderr_tail。建议 5-10s 间隔轮询。",
  { job_id: z.string() },
  async ({ job_id }) => { try { return T(await api("GET", `/jobs/${job_id}`)); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_golden_compare",
  "回归对账：把一批新账本和 golden 冻结基线逐条比对。默认 runs=runs_leju、golden=golden/leju_vendor（44 条）；另有 golden/openlet 基线 90 条。同步返回 {match, total, ok, diffs}。",
  { runs: z.string().default("runs_leju"), golden: z.string().default("golden/leju_vendor") },
  async (a) => { try { return T(await api("POST", "/golden_compare", a)); } catch (e) { return ERR(e); } }
);

server.tool(
  "organoid_artifact",
  "把作业产物下载到本地：receipt-*.json（验证数值细节）、keyframes、抓握力学曲线.png、对比视频 mp4 等。artifact_path 用 organoid_result 返回的 artifacts[].path；save_to 是本地保存路径。",
  {
    job_id: z.string(),
    artifact_path: z.string().describe("result.artifacts[].path 里的相对路径"),
    save_to: z.string().describe("本地保存路径（含文件名）"),
  },
  async ({ job_id, artifact_path, save_to }) => {
    try {
      const r = await fetch(`${API}/jobs/${job_id}/artifact?path=${encodeURIComponent(artifact_path)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
      const buf = Buffer.from(await r.arrayBuffer());
      const dest = path.resolve(save_to.replace(/^~(?=\/)/, os.homedir()));
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest, buf);
      return T({ saved: dest, bytes: buf.length });
    } catch (e) { return ERR(e); }
  }
);

server.tool(
  "organoid_upload",
  "把本地小数据集（目录或归档，打包后 ≤512MB）上传到服务器，返回服务器路径（可直接喂给 inspect/run/batch）。大数据集别用这个——请 rsync 到服务器后直接传路径。",
  { local_path: z.string().describe("本地数据集目录或 .tar.gz/.zip 文件") },
  async ({ local_path }) => {
    try {
      const src = path.resolve(local_path.replace(/^~(?=\/)/, os.homedir()));
      if (!fs.existsSync(src)) throw new Error(`local path not found: ${src}`);
      let file = src;
      let tmp = null;
      if (fs.statSync(src).isDirectory()) {
        tmp = path.join(os.tmpdir(), `organoid-up-${Date.now()}.tar.gz`);
        const tar = spawnSync("tar", ["-czf", tmp, "-C", path.dirname(src), path.basename(src)]);
        if (tar.status !== 0) throw new Error(`tar failed: ${tar.stderr}`);
        file = tmp;
      }
      const size = fs.statSync(file).size;
      if (size > 512 * 1024 * 1024) throw new Error(`archive is ${(size / 1e6).toFixed(0)}MB > 512MB cap；请改用 rsync`);
      const form = new FormData();
      form.append("file", new Blob([fs.readFileSync(file)]), path.basename(file));
      const r = await fetch(`${API}/upload`, { method: "POST", body: form });
      const text = await r.text();
      if (tmp) fs.unlinkSync(tmp);
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${text.slice(0, 400)}`);
      return T(JSON.parse(text));
    } catch (e) { return ERR(e); }
  }
);

// 同一段 guide 同时注册为 prompt（供支持 MCP prompts 的 client 给用户一键触发）
server.prompt("organoid_guide", "organoid 数据质检工作流指南（与 organoid_guide 工具同文）",
  () => ({ messages: [{ role: "user", content: { type: "text", text: GUIDE } }] }));

await server.connect(new StdioServerTransport());
