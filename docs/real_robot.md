# Phase 1 on the real Franka: lab guide

## Localization motion performance

The planned backend now separates collision/IK sampling (`cartesian_step: 0.002`)
from probe command spacing (`probe_step: 0.01`). Previously a sweep stopped and
restarted a JointPositionSkill every ~2 mm, paying the acceleration ramp and
controller termination buffer at each point. It now merges up to 10 mm of
checked probe motion into each skill. Every merged edge is independently checked
for shelf clearance, <=1 mm tip straightness error, and fixed tool orientation.
The whole ray is still checked before dispatch. Force detection runs throughout
the skill at the existing 20 ms polling interval and can stop before its endpoint.

The configured search/travel speeds, joint velocity/acceleration caps, contact
thresholds, gains and tracking tolerances are unchanged. This update retains
segment execution; it does not switch back to the streaming mode that previously
showed target freezes. Omit `probe_step` or set it to `0.002` to restore dense
probe commands. The permitted range is from `cartesian_step` up to the smaller
of `travel_step` and 20 mm.

Matched offline results using the same virtual plant and nominal placement:

| Sequence | Before | After | Joint commands before / after |
| --- | --- | --- | --- |
| Empty 245 mm sweep | 68.62 s | 30.16 s | 123 / 25 |
| 65 mm face probe, stopped on contact | 12.26 s | 5.70 s | 22 / 5 |
| 100 mm free-space travel | 6.38 s | 6.38 s | 3 / 3 |
| Full nominal localization with verification | 446.28 s | 293.60 s | 597 / 167 |

The full nominal run takes 34.2% less virtual time; both versions produced a
verified pose. Peak contact residual increased from 7.99 to 10.20 N in that
virtual run, below the existing 20 N safety limit. Fewer stops change the force
transient, so physical contact behavior must be checked on the robot. Peak joint
velocity and acceleration remain 0.15 rad/s and 0.3 rad/s². These are test-double
results, not hardware timings or measured robot forces. Representative benchmarks
explicitly enable path tracking checks; the existing hardware config's
`path_checks` choice is unchanged.

Validation: 54 tests passed across the executor, experiment harness, geometry and
touch probing. New tests stop a merged command on contact before its first
endpoint, enforce the polling/kinematic limits, and prevent dispatch after force
faults or early termination. Raw evidence is under `runs/real/motion-speed/`:
`before.json`, `after.json`, and `full-before/` / `full-after/` reports and audits.
The code/config update is local only; no robot motion was issued for this change.

```powershell
.venv/Scripts/python.exe scripts/real/benchmark_localization_motion.py --probe-step .002 --out runs/real/motion-speed/before.json
.venv/Scripts/python.exe scripts/real/benchmark_localization_motion.py --out runs/real/motion-speed/after.json
```

## Checked joint executor candidate (latest)

Hardware status (Nov 23): computer and arm are disconnected, confirmed by user.
The first startup trial stopped with about 10.8 mrad wrist endpoint error; the
subsequent trial also stopped early/short and its log remains on the lab host
(`20261122-235722-touch`). Its failed stage and final pose are unknown. The
20 mm free-air trial has not passed. Retrieve that log and inspect fresh robot
state after reconnecting before another authorized motion. Candidate q7
impedance is now 300 (previously 50); endpoint/safety tolerances are unchanged.
Local diagnostics now preserve termination timing, measured/desired joints and
controller flags before cleanup, including failed path audits. These diagnostics
have not yet been staged remotely.

Use `configs/real_planned.yaml` to select `robot.executor: planned_joint`.
The default `real.yaml` still selects the earlier Cartesian backend. The candidate
plans each complete ray from measured joints, certifies its interpolated joint
edges against the shelf/wall/table model, and commands `goto_joints` only.
It holds wrist yaw at pi and joint 3 at zero, monitors actual posture/clearance,
and disables Cartesian streaming/fallback. Adaptive paths are replanned from
feedback; hypothetical preflight contacts are never replayed on hardware.

Startup rotates the wrist at the current pose only above z=0.20 m, with <=2 mm
tip displacement and checked joint edges. From the recorded home this takes
about 40 s at the conservative 0.15 rad/s joint peak-speed limit. Low poses or
wrong starting joint-3 posture are refused rather than reset automatically.
Each subsequent command advances the planned tip <=2 mm. Minimum-jerk timing
bounds joint peak speed to 0.15 rad/s and acceleration to 0.3 rad/s^2. Both
measured `q` and the installed joint generator's actual start `q_d` are checked.

Limits: joint-3 deviation 0.01 rad, wrist/orientation deviation 0.03 rad,
tip tracking 4 mm, endpoint joint error 3 mrad for travel / 6 mrad for guarded
contact. The latter permits bounded elastic contact lag; it never permits an
early force-backstop termination to resume. All skills retain the force
backstop and virtual walls. Probe baselines persist across joint waypoints;
there is no per-waypoint or post-backstop re-zero. Sensor/controller faults,
stale feedback, e-stop modes, posture departure and interrupted execution stop
the owned skill and prevent the next joint command. Samples/audited paths are
saved with experiment logs.

Offline checks: full suite 83 passed, final executor-focused suite 24 passed;
full nominal localization replay through this executor passed in Docker's
Python 3.8 using a virtual joint/contact plant (no robot client). Nominal
replay used 1,486 deliberately slow joint commands / ~825 s virtual time.
This plant is a test double, not identified hardware dynamics. A nearly
tangential rotated-face case safely aborts at its tracking limit before the
force threshold; do not describe all placements as solved. The first hardware
step must be a short free-air tracking check, then a new face touch, before
another full localization. The hardware trials described above have not yet
validated that free-air tracking check.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_joint_executor.py -q
.venv/Scripts/python.exe scripts/real/test_joint_sequence.py --out runs/real/joint-executor-test --cases nominal
# Dry run: does not connect or move; --execute is deliberately omitted.
.venv/Scripts/python.exe scripts/real/phase1.py touch --config configs/real_planned.yaml --distance 0.02 --repeats 1
```

Results: `runs/real/joint-executor-release/summary.json` and `docker-replay.json`.
The new adapter's hardware I/O has been unit tested/compiled, but physical joint
tracking and force behavior remain unvalidated. Do not infer hardware success
from the offline nominal localization result.

## Offline full-sequence posture planning

Do not treat an endpoint-only clearance check as a validated probing sequence.
The offline planner uses the installed lab frankapy DH chain and all 12 robot
collision boxes (`configs/frankapy_geometry.json`), 30 shelf/support boxes with
10 mm inflation, the course virtual walls, and the reference tabletop. It
checks linear joint edges between waypoints with a conservative box-vertex
motion bound, in addition to IK endpoints. A separate MuJoCo check checks robot
self-collisions at the sampled configurations.

The raw DH tool orientation differs from reported `franka_tool` by a fixed
-45 degree z rotation. The planning model explicitly corrects the tool frame;
the collision-link transforms stay identical to the installed library. Both
are checked against a read-only robot snapshot and the independent MuJoCo chain.

The latest plan holds the finger-axis yaw at 180 degrees and joint 3 at zero.
The half-turn changes the wrist housing offset while preserving the peg axis
and symmetric gripper orientation. The earlier shelf-margin obstruction was
robot box 10 on link 7, at the right middle shelf; changing the elbow alone
cannot move that wrist box at a fixed full tool pose.

Reproduce the offline check (no robot execution option exists):

```powershell
.venv/Scripts/python.exe scripts/real/preflight_phase1.py --out runs/real/preflight-certified --variants --stress
.venv/Scripts/python.exe scripts/real/summarize_preflight.py runs/real/preflight-certified
```

Use a current read-only snapshot from `export_planning_state.py` inside Docker.
Ten cases cover a nominal box at (0.477,-0.106), four +/-5 mm / +/-5 degree
pose variants, no contact during the full sweep, +/-25 degree force-direction
bias, and +/-6 mm probing height. Contacts in this planner are geometric oracle
events, not a validation of real force sensing or motion tracking. Complete
commanded probe rays are checked even when the nominal contact stops them early.
Later adaptive branches are checked for the listed scenarios, not exhaustively
for every possible measured contact/force. Unreachable verification faces are
explicitly skipped and logged.

**Execution is not ready solely because this preflight passes.** The saved joint
paths require an executor that preserves their postures; the existing Cartesian
impedance `localize` command does not enforce the planned joint-3 value. Keep
`execution_ready: false` until that execution layer and a guarded hardware
tracking check are complete. `verify_preflight.py` independently checks saved
joint states with the installed frankapy in OFFLINE mode, without motion calls.

How to run the Phase 1 (box localization by touch) experiments on the course
robot with `scripts/real/phase1.py`. Rehearse every step offline first with
`--backend sim` (uses the MuJoCo scene; needs the project `.venv`).

## Control stack

| Backend | Status | Use when |
| --- | --- | --- |
| `frankapy` (default) | Rest sensing, table calibration, and repeated face touches measured on hardware Nov 22; full localization still pending | Normal course setup (User PC docker → Control PC server) |
| `pandapy` | Experimental, untested | Only with TA approval: needs frankapy's Control PC server stopped (libfranka allows one controller), a panda-py build matching the robot's system version, and running on a machine with a direct link to the robot |
| `sim` | Tested (28 unit tests, 13 hard box poses) | Offline rehearsal, CI |

Why frankapy first: it is what the lab is set up for, and its 1 kHz impedance
controller runs on the Control PC. Contact detection runs in Python at about
50 Hz on `get_ee_force_torque()` (libfranka's `O_F_ext_hat_K`, base frame).
At 10 mm/s that is about 0.2 mm of travel per sample. Native probes stop on a
debounced force jump, with frankapy force thresholds as an additional backstop;
these do not guarantee a peak contact-force bound. panda-py would read the
same force estimate (plus per-joint `tau_ext_hat_filtered`) at 1 kHz: better,
but not needed unless the `noise`/`touch` experiments show the 50 Hz loop is
too slow or noisy.

## One-time setup (User PC)

1. Follow Lab 1 to power on, unlock (blue light), start the Control PC server
   and open a docker terminal.
2. Put this repo in your team folder:
   `~/16662_RobotAutonomy/src/devel_packages/<team>/` (git clone or copy).
3. Inside the docker: `python3 -c "import numpy, scipy, yaml, frankapy"`.
   Install anything missing with `pip3 install --user scipy pyyaml`.
   MuJoCo is not needed for the real robot.

Safety (course rules): never alone; a partner's hand on the e-stop for every
`--execute`; stay outside the table area. Every moving command asks you to type
`RUN`; anything without `--execute` is a dry run. Probes abort above
`contact.safety_force` (20 N). Targets outside the search rectangle (+3 cm) are
refused.

## Session plan

Run from the repo root inside the docker. Each run logs to `runs/real/<time>-<command>/`
(`result.json` + `samples.npz`).

| Step | Command | What to record |
| --- | --- | --- |
| 1 | `python3 scripts/real/phase1.py check` | Loop rate (aim ~50 Hz) and force noise at rest. No motion. |
| 2 | `... grasp --execute` | Peg in the gripper, shaft down. |
| 3 | Move the tip ~5 cm above an empty table spot (guide mode or `--start`) | |
| 4 | `... table --execute` | Put the printed `table_z` into `configs/real.yaml`; flip `robot.force_sign` if it says so. |
| 5 | `... noise --start X Y Z --execute` (Z ~ table_z + 0.05, clear path) | Set `contact.force_threshold` to the suggested value (or higher). |
| 6 | Tape the box down. `... touch --start X Y Z --direction 1 0 --repeats 5 --execute` with the tip ~4 cm from a face at `table_z + 0.025` | Contact spread (aim < 1 mm), peak force, force direction. Repeat from 2-3 sides; check the peg did not slip in the grasp. |
| 7 | `... groundtruth --execute` (add `--corners` for the box pose) | Guide the peg into the hole: hole truth. |
| 8 | `... localize --truth runs/real/groundtruth.json --execute` | Box centre / yaw error, hole-estimate error, time. Repeat with the box moved (5-10 poses). |
| 9 | `... refit --run runs/real/<...>-localize` | Offline: initial 4-contact fit vs. verified fit, other force weights. |

Things to check in the outputs:
- `force_opposes_motion=False` on a contact: the force sign is wrong (step 4).
- Many `stalled` probes: the arm cannot follow the target (reach or joint limits);
  shrink `workspace.search_corners`.
- Safety stops while probing: raise the threshold only if `noise` supports it;
  otherwise lower `motion.search_speed`.
- `localize` aligns the fingers with probing motion when clearance permits,
  falling back to a perpendicular wrist orientation near a wall. Check the peg
  for slip especially on perpendicular probes.

## Hardware findings and wall clearance (Nov 22)

Shelf geometry comes from the user's Lab 2 `team_kilo.zip`, and matches all
30 shelf/support boxes in the simulation asset. `configs/shelf_obstacles.json`
records the boxes and their provenance. Board bounds were measured; support
shapes are estimates. The real backend checks the full arm against these boxes
with a 10 mm margin before and during motion. The hand planner additionally
keeps its entire planar envelope between shelf inner edges y=-0.32 and +0.30
at all experiment heights. Outer virtual walls remain enabled.

Search bounds are now x=[0.32,0.565], y=[-0.21,0.15]. The first hardware search
reached two contacts but could not plan the next wrist orientation. The planner
now tries wall-safe cardinal orientations as additional fallbacks. Positioning
descents must reach within 6 mm of search height with >=5 mm finger clearance;
otherwise localization stops before collecting side evidence. Verification
skips unplannable faces and labels unverified/ambiguous fits explicitly.

Free-space travel uses static Cartesian impedance skills with gains
[1500,1500,1500,100,100,100], a 6 s minimum duration, 20 mm/s average travel
speed, and force backstops. A 6 s high-clearance trial reached within 3.4 mm
of its target. Internal Cartesian returns failed with joint acceleration
discontinuity even at 10 mm/s; slowing alone did not fix those returns.
Native guarded forward probing remains on the internal controller: a 117 mm
free-air test had no false contacts and a 2.14 N peak tracking residual.
Real-contact validation is still pending for the drift-tracking detector.
An internal-controller face-touch attempt was also rejected immediately with
joint acceleration discontinuity. Native probes now select static impedance
via `motion.native_probe_controller: impedance`; this combination still needs
hardware contact validation. The Python detector and force backstop stay enabled.

The measured table tip height is 0.0072 m; probing at 0.0322 m leaves the
fingers about 12 mm above the box top. Do not use the old 15 mm search offset:
the fingers can land on the box before the peg shaft touches its side.

Use `motion.frankapy_probe_mode: native`. Streamed targets stopped tracking
after about 23 mm in the lab. Starting a native skill shifts the force estimate
about 2.5 N; free-air movement adds drift of roughly 0.2 N/mm. The detector now
settles for 0.25 s and tracks a baseline over 5 mm of actual tip travel, then
declares contact on a 4 N jump held for three samples. Deterministic synthetic
traces test this logic, but the updated detector still needs hardware validation.
The `noise` command records free-air forces; to exercise contact detection in
free air, use a single `touch` probe at a verified clear height instead.

Keep frankapy's virtual walls enabled. Measured planar inner faces are
x in [-0.405, 0.745] and y in [-0.455, 0.455] m. The added hand guard uses
conservative half extents of 0.165 m along the fingers and 0.095 m across,
plus a 0.01 m margin, from the peg axis. It checks translation and the full
shortest wrist rotation. This is a planar approximation, not full-arm collision
checking; frankapy remains responsible for its arm collision boxes.

At x=0.62 m only the perpendicular finger orientation fits. The localizer
rotates at a safe starting point before translating, and refuses motion if
neither orientation clears the walls. The backend applies the guard to direct
motion commands too. A current pose outside the guard requires manual recovery;
the guard never bypasses virtual walls to escape.

Before each motion step, confirm a partner is at the e-stop and the area is
clear. Before connecting a client, check `rosnode list` for another
`franka_arm_client`. If wall contact has left all moves refused, recover by
hand and restart the Control PC server as described in `Agent_handoff.md`.

## What the localizer does

1. Snake sweep of `workspace.search_corners` at `table_z + search_offset` until the
   peg's side touches the box.
2. Cross from the far side along the contact force (opposite face), then probe both
   ways along the perpendicular through the midpoint.
3. Fit the box pose to the 4 contacts (`kilo/localization/fit.py`).
4. Verification (up to `fit.verify_rounds`): probe straight at each fitted face
   centre and refit with all contacts. Four contacts can fit two different poses
   (`docs/experiments/phase1_refit*.json`); in simulation verification corrected
   every such case among 13 hard seeds (12/13 within 0.7 mm and 2.4 deg; seed 84
   still fails on the simulator's weak controller at full reach).

Every descent to probing height is force-guarded, so landing on the box top
stops and is recorded (and counts as evidence against the current fit).

## Rehearse offline

```bash
.venv/Scripts/python.exe scripts/real/phase1.py localize --backend sim --seed 8
.venv/Scripts/python.exe scripts/real/phase1.py touch --backend sim --start 0.36 -0.06 0.0322 --direction 1 0
.venv/Scripts/python.exe scripts/real/sim_check_localizer.py --seeds 0 8 49 94 --workers 4
```

The sim backend adds 2 N of random bias and 0.3 N of noise to the force reading.
Its impedance controller (borrowed from `kilo/localization/probenav.py`) is
softer (200 N/m) than the real setting (600 N/m) and can stall at full reach.
