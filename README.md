# WorldLedger

**A verification-centered architecture for embodied agents**

WorldLedger organizes embodied-agent workflows around context-bound proposals, explicit evidence, replayable execution, and a separate acceptance authority. The public release contains the architecture report, a Chinese translation, selected implementation modules, compact case summaries, and reproducibility notes.

The report focuses on general engineering principles: binding an action to the body, scene, initial state, task, and protocol; separating proposal ranking from acceptance; preserving rejected and unevaluated evidence; and compiling verified trajectories into data views with explicit timing and label semantics.

## Contents

- `report/` — English and Chinese technical reports, six paired vector figures, their build script, and a public claim-source index.
- `organoid_kernel/` — the Python package containing evidence, validation, trajectory, and replay modules.
- `scripts/` — selected model-fetch, trajectory, verification, and replay entry points.
- `examples/` — compact, path-free summaries for the cited engineering cases.
- `docs/` — architecture, transaction boundary, multi-robot pipeline, and public-scope notes.
- `mcp/` — optional HTTP job service and local stdio MCP client skeleton.

## Quick start

Create a Python environment and install the core dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_adapters_synthetic.py tests/test_gate_fixes.py -q
```

For the fixed-base trajectory example, fetch the pinned upstream robot assets and run a small batch:

```bash
python scripts/fetch_robot_models.py --robot all
python scripts/build_multi_robot_dataset.py --families 2 --workers 1 --seed 20260911 --skip-training --out /tmp/worldledger-demo
python scripts/verify_multi_robot_dataset.py /tmp/worldledger-demo --replay-all --workers 1
```

The generated output is a local development artifact. Critic training requires a separate PyTorch installation; this example skips it. Simulation results are not hardware measurements, and a critic score never authorizes execution.

The dataset verification worker is included. The full stateful transaction controller is an external component, and the teacher-data and audiovisual cases provide summaries rather than complete runnable archives. The report's system diagram and module appendix describe these boundaries.

## Scope and privacy

The public repository intentionally excludes personal information, home directories, internal hostnames and IP addresses, deployment credentials, raw production archives, and uncleared third-party assets. The excluded material remains in the original working directory and is not represented as a public dataset. See [`docs/public-scope.md`](docs/public-scope.md).

## Report and DOI status

Read the [English report](report/technical-report.md) or [中文版](report/technical-report-zh.md); downloadable PDFs and figure build instructions are listed in [report/README.md](report/README.md). No DOI has been reserved or registered yet. Author metadata, rights review, and the final public license must be completed before a Zenodo record is published.
