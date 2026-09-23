# Multi-Robot Trajectory Pipeline

## Scope

The pipeline has three model-backed robot profiles: the existing Kuavo
`biped_s200049`, plus `franka_emika_panda` and `universal_robots_ur5e`.
The two new profiles use pinned MuJoCo Menagerie assets, including the original
MJCF, visual meshes, collision geometry, inertial parameters, actuator limits,
source README, and license. The upstream descriptions are simplified models
derived from vendor URDFs, not hardware calibrations.

Panda: Apache-2.0. UR5e: BSD-3-Clause. Source commit:
`8161bba264d7fa7c99ca301e91e7fb44737676ad`.
Repository: <https://github.com/google-deepmind/mujoco_menagerie>.
Each local `assets/robots/<profile>/source-manifest.json` records hashes.

This is fixed-base simulation and trajectory tooling. It does not connect to
hardware, certify physical safety, provide calibrated velocity limits, or claim
arbitrary-robot generalization. Fourier remains a separate measured-data adapter;
its unnamed channels are not guessed into these robot profiles.

## Run

From the repository root:

```bash
python3 scripts/fetch_robot_models.py
python3 scripts/build_multi_robot_dataset.py \
  --families 60 --workers 4 --seed 20260911 --out runs_multi_robot_480_v1

python3 scripts/build_multi_robot_dataset.py --replay \
  runs_multi_robot_480_v1/episodes/family_0001/universal_robots_ur5e/raised

python3 scripts/verify_multi_robot_dataset.py \
  runs_multi_robot_480_v1 --replay-all --workers 4

python3 scripts/score_robot_candidate.py \
  runs_multi_robot_480_v1/feasibility_critic.pt \
  runs_multi_robot_480_v1/episodes/family_0001/universal_robots_ur5e/raised/record.json
```

Nonempty output directories are refused. A full batch is 60 parameter families,
two robot profiles, and four candidates per robot: 480 episodes. Installation
needs network access; generation uses local assets. NumPy, SciPy, MuJoCo and
PyTorch are required. Video rendering additionally uses OpenCV and
`imageio-ffmpeg`.

## Tasks And Randomization

- `reach_target`: reach an XYZ target, no orientation requirement.
- `obstacle_transfer`: move the TCP around a fixed obstacle. It carries no object.
- `inspection_sweep`: visit three XYZ targets in order. It does not measure
  camera coverage or recognize inspected objects.

Per family, initial joint perturbation, target XYZ offset, obstacle XYZ offset,
obstacle half-size and control noise are randomized. The same family is applied
to both robots relative to each model's initial TCP. Four bounded candidate
templates vary route height and duration: direct, raised, fast, high-clearance.
The deliberately aggressive 0.65 m high-clearance candidate supplies many IK
solver failures; this makes part of the IK classification problem easy. Do not
interpret its accuracy as broad feasibility understanding.

Candidates in a robot scene share identical initial integration state and
scene geometry. Candidate correction does not move obstacles or alter target
geometry to obtain a pass. Control noise is added before `mj_step`; exact
applied commands are recorded.

## Episode Contract

`organoid.trajectory.v2` consists of `trajectory.npz` and `trajectory.json`.
All arrays have shapes, dtypes, units, provenance, and an explicit clock or
`clock: null` for static data.

Simulation states are at 500 Hz with N+1 samples; actions have N samples and
apply over the following 2 ms interval. Float64 dynamics are preserved.
Arrays include full `qpos/qvel`, named arm joint position and velocity,
TCP position, applied actuator controls, commanded arm references, actuator
forces, contact flags/depth, integration initial state, task targets,
candidate waypoints, and pre-execution critic features.

`joint_reference` is in radians. Raw Panda controls also include the source
model's 0-255 gripper control, so `control` must not be interpreted as a vector
of joint angles. The files retain native actuator names and source model.
All sensor-like channels in this batch are simulated, not measured sensors.

Other episode files:

- `scene.xml`: exact model and scene used by the episode.
- `receipt.json`: evaluated checks, stop reason, labels, scope, replay errors.
- `record.json`: identity, family, task, features, labels, provenance hashes.

Every episode, including rejected partial trajectories, is replayed from the
saved integration state with the saved control sequence. State, actuator force,
TCP trajectory, and solver warnings are checked. A failed replay masks all
learning labels. Current scene XML uses absolute mesh paths; this output is a
local development dataset, not a relocatable distribution archive.

Top-level files include protocol, scene specs, records, summary, split,
critic checkpoint, per-example predictions, evaluation, and source snapshot.
Each `summary.json` correction pair identifies the rejected direct candidate
and a physically accepted alternative with matching initial-state and scene
hashes. This is fixed-order candidate search, not evidence of learned repair.

## Label Semantics

`success=true` requires the entire path to complete, ordered targets and final
position to pass the 25 mm threshold, no modeled obstacle/self collision above
0.1 mm penetration, modeled joint/effort constraints, finite dynamics, and no
external forces. These are task-specific simulation checks, not hardware safety.

`collision_free=false` records a witnessed collision. An IK stop without a
witnessed collision leaves the unexecuted suffix unknown, so this label is null.
`ik_feasible=false` records the bounded solver exceeding its 2 mm tolerance.
An early collision leaves later IK untested, so that label is null. Solver
failure is not a proof of global geometric infeasibility.
`stable_grasp` is null for all three tasks, because they do not perform a grasp.
Nulls are excluded from each head's loss and metrics; no zero-fill labels.

## Critic And Evaluation

The new critic takes a fixed whitelist of 36 pre-execution features: initial
arm state and joint mask, TCP start, goal and obstacle geometry, and candidate
path descriptors. It does not consume final error, measured contacts, IK
residuals, slip, or other rollout outcomes. Path descriptors currently support
the four specified candidate templates, not arbitrary unseen path geometry.

The split is by the entire parameter family, including both robots and every
candidate. Normalization uses only the training split; early stopping uses
validation only. Metrics include per-head confusion counts, balanced accuracy,
Brier score, training-prior baseline and same-scene candidate ranking.
Additional models train on one robot and test on the other, on held-out
families. These test robot generalization separately from scene generalization.

`torch.load(..., weights_only=True)` is supported. Unknown profiles and features
outside a bounded training envelope return `not_evaluated_ood`. This simple
envelope is not a complete OOD detector. Scores always return
`execution_authorized=false`; MuJoCo remains the acceptance authority.

The older `runs_embodied_training_100_v2` checkpoint included post-rollout
metrics in its inputs. Its reported 93.18% must not be used as pre-execution
prediction evidence. Older pushes were also affected by a default-goal change
and incorrect barrier-height scaling, now covered by regression tests.
Those historical files are preserved; the new batch is independently generated.

## External Joint States

Use `scripts/robot_trajectory.py import <file.npz> --sidecar <schema.json>
--out <new-directory>`.

The sidecar must explicitly declare:

```json
{
  "profile": "universal_robots_ur5e",
  "joint_names": [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
  ],
  "joint_unit": "rad",
  "time_unit": "s",
  "state_semantics": "joint_position",
  "origin": "observed",
  "time_key": "timestamps",
  "state_key": "joint_positions"
}
```

Joint columns may be reordered; their names, units and timestamp ordering are
validated. This import audits observed positions and model limits, not dynamics.
Unknown robot mappings are rejected. It does not turn observed state into
commands or fill missing force measurements. Use `origin: simulated` for
simulation sources.

Unified trajectories also enter `organoid_kernel.cli inspect/run`. The old
URDF-only physics validators remain explicitly `not_evaluated` for the new
MJCF profiles; use the dedicated saved-control replay command for this batch.
This prevents an MJCF profile name alone from falsely enabling legacy checks.

## Task-Space Transfer

```bash
python3 scripts/robot_trajectory.py transfer \
  runs_multi_robot_480_v1/episodes/family_0000/franka_emika_panda/direct \
  --robot universal_robots_ur5e \
  --out runs_multi_robot_transfers_v1/panda_to_ur5e_reach
```

The current transfer contract consumes declared metre-valued Cartesian
waypoints, task targets and initial TCP in model-world axes. It translates the
task origin to the destination robot's initial TCP, keeps scale/axes explicit,
re-solves IK, reruns destination physics, and records the transform and source
hash. It does not copy actuator commands, preserve source timing exactly,
transfer orientation/grasp/contact semantics, or infer camera extrinsics.

The initial demonstrations include one accepted Panda-to-UR5e reach and one
rejected UR5e-to-Panda obstacle path. Both have independently verified replay.
Broader transfer needs more robot morphologies, harder candidate distributions,
explicit object/contact tasks and measured calibration, not just more copies
of these path templates.

## Transaction Control Plane

The new pipeline connects to the actual local transaction control plane HTTP/MCP implementation.
Robot assets, initial integration state, scene, targets, action and protocol are
hash-bound. The critic only ranks; each verification launches fresh verification kernel
dynamics plus saved-control replay. transaction control plane independently gates transactional
acceptance and preserves rejected and unknown attempts.

See [transaction control boundary](transactional-verification.md) for APIs, scope, evidence,
local reproduction and deployment. Updating local source does not upgrade an
already running remote MCP service.

## Regression Tests

```bash
python3 -m pytest tests/test_multi_robot.py tests/test_humanoid_tasks.py \
  tests/test_humanoid_frozen_evidence.py -q
```

Tests cover profile mapping, unit rejection, timestamps, missingness, hashes,
whole-family split, input leakage, failure masking, original Kuavo regressions,
native-model replay, and task-space transfer.
