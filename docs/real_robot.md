# Phase 1 on the real Franka: lab guide

How to run the Phase 1 (box localization by touch) experiments on the course
robot with `scripts/real/phase1.py`. Rehearse every step offline first with
`--backend sim` (uses the MuJoCo scene; needs the project `.venv`).

## Control stack

| Backend | Status | Use when |
| --- | --- | --- |
| `frankapy` (default) | Written against frankapy's documented API and its `run_dynamic_pose.py` example; **not yet run on the robot** | Normal course setup (User PC docker → Control PC server) |
| `pandapy` | Experimental, untested | Only with TA approval: needs frankapy's Control PC server stopped (libfranka allows one controller), a panda-py build matching the robot's system version, and running on a machine with a direct link to the robot |
| `sim` | Tested (28 unit tests, 13 hard box poses) | Offline rehearsal, CI |

Why frankapy first: it is what the lab is set up for, and its 1 kHz impedance
controller runs on the Control PC. Contact detection runs in Python at about
50 Hz on `get_ee_force_torque()` (libfranka's `O_F_ext_hat_K`, base frame).
At 10 mm/s that is about 0.2 mm of travel per sample, and the push is capped
at about 2x the threshold even if detection is late. panda-py would read the
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
| 6 | Tape the box down. `... touch --start X Y Z --direction 1 0 --repeats 5 --execute` with the tip ~4 cm from a face at `table_z + 0.015` | Contact spread (aim < 1 mm), peak force, force direction. Repeat from 2-3 sides; check the peg did not slip in the grasp. |
| 7 | `... groundtruth --execute` (add `--corners` for the box pose) | Guide the peg into the hole: hole truth. |
| 8 | `... localize --truth runs/real/groundtruth.json --execute` | Box centre / yaw error, hole-estimate error, time. Repeat with the box moved (5-10 poses). |
| 9 | `... refit --run runs/real/<...>-localize` | Offline: initial 4-contact fit vs. verified fit, other force weights. |

Things to check in the outputs:
- `force_opposes_motion=False` on a contact: the force sign is wrong (step 4).
- Many `stalled` probes: the arm cannot follow the target (reach or joint limits);
  shrink `workspace.search_corners`.
- Safety stops while probing: raise the threshold only if `noise` supports it;
  otherwise lower `motion.search_speed`.
- `localize` keeps the wrist at one angle by default (`motion.align_wrist: false`)
  to stay clear of joint-7 limits. The peg is round, so this does not change
  contact geometry; step 6 checks the grasp holds.

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
.venv/Scripts/python.exe scripts/real/phase1.py touch --backend sim --start 0.36 -0.06 0.016 --direction 1 0
.venv/Scripts/python.exe scripts/real/sim_check_localizer.py --seeds 0 8 49 94 --workers 4
```

The sim backend adds 2 N of random bias and 0.3 N of noise to the force reading.
Its impedance controller (borrowed from `kilo/localization/probenav.py`) is
softer (200 N/m) than the real setting (600 N/m) and can stall at full reach.
