#!/usr/bin/env python3
"""organoid_service — WorldLedger verification kernel的 HTTP job 服务。

形态与 musicodec 后端同款：submit → job_id → poll。
部署布局（服务器 ~/organoid_service/）：
    app.py  start.sh  venv/  kernel/（内核目录）  jobs/<id>/  uploads/<sha>/
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
KERNEL = Path(os.environ.get("WORLDLEDGER_KERNEL", BASE.parents[1])).resolve()
JOBS = BASE / "jobs"
UPLOADS = BASE / "uploads"
JOBS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
PY = sys.executable
SERVICE_VERSION = "0.1.0"
UPLOAD_CAP = 512 * 1024 * 1024
REPORT_CAP = 120_000

app = FastAPI(title="organoid_service", version=SERVICE_VERSION)
pool = ThreadPoolExecutor(max_workers=2)
EXPERIMENT_LOCK = threading.Lock()

EXPERIMENTS = {
    "grasp_transplant": {"script": "scripts/grasp_transplant.py", "outdir": "runs_grasp"},
    "fullbody_replay": {"script": "scripts/fullbody_replay.py", "outdir": "runs_fullbody"},
}

# 绕过 CLI inspect 的 4000 字符截断：直接库调打印完整 summary
INSPECT_SNIPPET = r"""
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from organoid_kernel.cli import _load_episode
pkg = _load_episode(Path(sys.argv[1]), int(sys.argv[2]))
print(json.dumps(pkg.summary(), ensure_ascii=False, default=str))
"""


def sub_env():
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    return env


def resolve_dataset(p: str) -> Path:
    """数据集路径：绝对路径 / ~ 展开原样用；相对路径相对内核根（samples/... 即内置样例）。"""
    path = Path(os.path.expanduser(p))
    if not path.is_absolute():
        path = KERNEL / path
    if not path.exists():
        raise HTTPException(404, f"dataset path not found on server: {path}")
    return path


def _write_status(jobdir: Path, **kw):
    p = jobdir / "status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(kw)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    return st


def _ledger_digest(ledger_path: Path) -> Optional[dict]:
    if not ledger_path.exists():
        return None
    led = json.loads(ledger_path.read_text())
    return {
        "episode_id": led.get("episode_id"),
        "grade": led.get("grade"),
        "grade_reason": led.get("grade_reason"),
        "policy": led.get("policy"),
        "identity": (led.get("identity") or {}).get("status"),
        "profile": (led.get("identity") or {}).get("profile"),
        "claims": [
            {"claim": c.get("claim"), "status": c.get("status"), "reason": c.get("reason")}
            for c in (led.get("claims") or [])
        ],
        "repairs": led.get("repairs"),
        "warnings": led.get("warnings"),
    }


def _post_run(jd: Path) -> dict:
    return {"result": _ledger_digest(jd / "out" / "claim-ledger.json")}


def _post_batch(jd: Path) -> dict:
    eps = []
    for d in sorted((jd / "out").glob("ep*")):
        dig = _ledger_digest(d / "claim-ledger.json")
        eps.append({"ep": d.name, "grade": (dig or {}).get("grade"),
                    "grade_reason": (dig or {}).get("grade_reason")})
    grades: dict = {}
    for e in eps:
        grades[str(e["grade"])] = grades.get(str(e["grade"]), 0) + 1
    summary = {"n_episodes": len(eps), "grades": grades, "episodes": eps}
    (jd / "out" / "batch-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    return {"result": summary}


def _make_post_experiment(outdir: str):
    def post(jd: Path) -> dict:
        src = KERNEL / outdir
        dst = jd / "out"
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        j = next(iter(sorted(dst.glob("*.json"))), None)
        res = json.loads(j.read_text()) if j else None
        return {"result": {"summary_json": j.name if j else None, "detail": res}}
    return post


def _exec_job(job_id: str, cmd: List[str], serialize: bool, post):
    jd = JOBS / job_id
    lock = EXPERIMENT_LOCK if serialize else None
    if lock:
        lock.acquire()
    try:
        _write_status(jd, status="running", started=time.time())
        with open(jd / "stdout.log", "w") as so, open(jd / "stderr.log", "w") as se:
            rc = subprocess.run(cmd, cwd=KERNEL, stdout=so, stderr=se, env=sub_env()).returncode
        extra = {}
        try:
            if post and rc == 0:
                extra = post(jd) or {}
            status = "done" if rc == 0 else "failed"
        except Exception as e:
            status, extra = "failed", {"post_error": repr(e)}
        _write_status(jd, status=status, rc=rc, finished=time.time(), **extra)
    finally:
        if lock:
            lock.release()


def _submit(kind: str, cmd: List[str], serialize=False, post=None, meta=None) -> str:
    job_id = uuid.uuid4().hex[:12]
    jd = JOBS / job_id
    jd.mkdir()
    _write_status(jd, job_id=job_id, kind=kind, status="queued",
                  cmd=" ".join(map(str, cmd)), created=time.time(), **(meta or {}))
    pool.submit(_exec_job, job_id, cmd, serialize, post)
    return job_id


# ---------------- request models ----------------

class InspectReq(BaseModel):
    path: str
    episode: int = 0


class RunReq(BaseModel):
    path: str
    episode: int = 0
    policy: Optional[str] = None
    profile: Optional[str] = None
    skip: List[str] = []


class BatchReq(BaseModel):
    path: str
    limit: Optional[int] = None
    policy: Optional[str] = None
    profile: Optional[str] = None
    skip: List[str] = ["visual"]


class ExperimentReq(BaseModel):
    name: str


class GoldenReq(BaseModel):
    runs: str = "runs_leju"
    golden: str = "golden/leju_vendor"


# ---------------- endpoints ----------------

@app.get("/health")
def health():
    deps = {m: find_spec(m) is not None for m in
            ["numpy", "pandas", "pyarrow", "mujoco", "cv2", "h5py", "zarr",
             "rosbags", "mediapipe", "matplotlib", "PIL"]}
    return {"ok": True, "kernel_present": (KERNEL / "organoid_kernel" / "cli.py").exists(),
            "deps": deps, "n_jobs": len(list(JOBS.iterdir())),
            "versions": {"service": SERVICE_VERSION, "python": sys.version.split()[0]}}


@app.post("/inspect")
def inspect(req: InspectReq):
    path = resolve_dataset(req.path)
    r = subprocess.run([PY, "-c", INSPECT_SNIPPET, str(path), str(req.episode)],
                       cwd=KERNEL, capture_output=True, text=True, timeout=180, env=sub_env())
    if r.returncode != 0:
        raise HTTPException(500, f"inspect failed: {r.stderr[-1500:]}")
    try:
        return {"summary": json.loads(r.stdout)}
    except json.JSONDecodeError:
        return {"summary_raw": r.stdout[-8000:]}


@app.post("/run")
def run(req: RunReq):
    path = resolve_dataset(req.path)
    job_id = uuid.uuid4().hex[:12]  # 占位以便 --out 指向 job 目录
    jd = JOBS / job_id
    jd.mkdir()
    cmd = [PY, "-m", "organoid_kernel.cli", "run", str(path),
           "--episode", str(req.episode), "--out", str(jd / "out")]
    if req.policy:
        cmd += ["--policy", req.policy]
    if req.profile:
        cmd += ["--profile", req.profile]
    if req.skip:
        cmd += ["--skip", *req.skip]
    _write_status(jd, job_id=job_id, kind="run", status="queued",
                  cmd=" ".join(cmd), created=time.time(), dataset=str(path))
    pool.submit(_exec_job, job_id, cmd, False, _post_run)
    return {"job_id": job_id, "hint": "poll /jobs/{job_id}; 单条全链约 1-5s，含 hand_video 时 5-30s"}


@app.post("/batch")
def batch(req: BatchReq):
    path = resolve_dataset(req.path)
    job_id = uuid.uuid4().hex[:12]
    jd = JOBS / job_id
    jd.mkdir()
    cmd = [PY, "-m", "organoid_kernel.cli", "batch", str(path), "--out", str(jd / "out")]
    if req.limit:
        cmd += ["--limit", str(req.limit)]
    if req.policy:
        cmd += ["--policy", req.policy]
    if req.profile:
        cmd += ["--profile", req.profile]
    if req.skip:
        cmd += ["--skip", *req.skip]
    _write_status(jd, job_id=job_id, kind="batch", status="queued",
                  cmd=" ".join(cmd), created=time.time(), dataset=str(path))
    pool.submit(_exec_job, job_id, cmd, False, _post_batch)
    return {"job_id": job_id, "hint": "poll /jobs/{job_id}; 耗时 ≈ 每条 0.2-1.2s × episode 数"}


@app.post("/experiment")
def experiment(req: ExperimentReq):
    if req.name not in EXPERIMENTS:
        raise HTTPException(400, f"unknown experiment: {req.name}; choices: {list(EXPERIMENTS)}")
    spec = EXPERIMENTS[req.name]
    job_id = _submit(f"experiment:{req.name}", [PY, spec["script"]],
                     serialize=True, post=_make_post_experiment(spec["outdir"]))
    return {"job_id": job_id, "hint": "poll /jobs/{job_id}; 约 40-60s，实验作业全局串行"}


@app.post("/golden_compare")
def golden_compare(req: GoldenReq):
    runs = resolve_dataset(req.runs)
    golden = resolve_dataset(req.golden)
    r = subprocess.run([PY, "-m", "organoid_kernel.cli", "golden-compare", str(runs), str(golden)],
                       cwd=KERNEL, capture_output=True, text=True, timeout=300, env=sub_env())
    m = re.search(r"golden 对账: (\d+)/(\d+) 一致", r.stdout)
    match, total = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return {"match": match, "total": total,
            "ok": match == total if m else False,
            "stdout": r.stdout[-6000:], "stderr": r.stderr[-1500:] or None}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    jd = JOBS / job_id
    if not jd.exists():
        raise HTTPException(404, "job not found")
    st = json.loads((jd / "status.json").read_text())
    if st.get("status") in ("done", "failed"):
        rep = jd / "out" / "quality-report.md"
        if rep.exists():
            st["report_md"] = rep.read_text()[:REPORT_CAP]
        arts = []
        for f in sorted(jd.rglob("*")):
            if f.is_file() and f.name != "status.json":
                arts.append({"path": str(f.relative_to(jd)), "bytes": f.stat().st_size})
            if len(arts) >= 400:
                break
        st["artifacts"] = arts
        errlog = jd / "stderr.log"
        if errlog.exists():
            tail = errlog.read_text()[-2000:]
            st["stderr_tail"] = tail or None
    return st


@app.get("/jobs/{job_id}/artifact")
def job_artifact(job_id: str, path: str):
    jd = (JOBS / job_id).resolve()
    if not jd.exists():
        raise HTTPException(404, "job not found")
    f = (jd / path).resolve()
    if not str(f).startswith(str(jd) + os.sep) or not f.is_file():
        raise HTTPException(404, f"artifact not found: {path}")
    return FileResponse(f)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path):
    for m in tf.getmembers():
        p = (dest / m.name).resolve()
        if not str(p).startswith(str(dest.resolve())):
            raise HTTPException(400, f"unsafe path in archive: {m.name}")
    tf.extractall(dest)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp = UPLOADS / f".tmp-{uuid.uuid4().hex[:8]}"
    sha = hashlib.sha256()
    n = 0
    with open(tmp, "wb") as w:
        while chunk := await file.read(1 << 20):
            n += len(chunk)
            if n > UPLOAD_CAP:
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {UPLOAD_CAP >> 20}MB cap; 大数据请 rsync 到服务器后直接传路径")
            sha.update(chunk)
            w.write(chunk)
    dest = UPLOADS / sha.hexdigest()[:12]
    if dest.exists():
        tmp.unlink()
        return {"path": str(dest), "bytes": n, "cached": True}
    dest.mkdir()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(tmp) as z:
                z.extractall(dest)
        else:
            with tarfile.open(tmp) as t:
                _safe_extract_tar(t, dest)
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(400, f"cannot extract archive: {e!r}")
    finally:
        tmp.unlink(missing_ok=True)
    inner = list(dest.iterdir())
    path = inner[0] if len(inner) == 1 and inner[0].is_dir() else dest
    return {"path": str(path), "bytes": n, "cached": False}
