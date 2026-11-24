# Running Phase 1 localization yourself

These scripts run on the **lab User PC** (iam-camilo), in a normal terminal, *outside* docker.
They wrap `scripts/real/phase1.py` inside the `frankapy_docker` container.

```bash
cd ~/16662_RobotAutonomy/src/devel_packages/kilo_project/scripts/real/lab
```

| Step | Command | Moves the robot? |
|---|---|---|
| 0. Status: server, errors, pose, gripper, other clients | `./status.sh` | no |
| 1. Arm to home (shelf-checked joint move, 8 s) | `./home.sh` | yes, type RUN |
| 2. Grip the peg (open, place peg by its square top, close at 40 N) | `./grasp.sh` | gripper only |
| 3. Localize: run, log, then charts | `./localize.sh` | yes, type RUN |
| 4. Re-make charts for a run | `./view.sh [run] [--open]` | no |
| Offline rehearsal, no robot | `./rehearse.sh [scenario ...]` | no |

Before any motion: someone at the e-stop, the table area clear, no guide-mode or other arm
script running (the scripts refuse to start if they see one). The Control PC server must be up
(`./status.sh` checks it and prints the start command if not).

## What a run writes

`runs/real/<stamp>-localize/`

- `result.json`: config, contacts, events, estimate, and a per-path motion audit.
- `samples.npz`: force and tip traces for every probe.
- `console.log`: everything printed during the run.
- `charts/report.txt`:
  - outcome, including any SAFETY STOP and its controller error flags;
  - the estimate, and the score against truth if `--truth` was given;
  - every contact;
  - the estimate after each contact;
  - the events.
- `charts/1_overview.png`: top-down map of the search area, peg-tip paths, contacts with
  measured-force arrows, the initial and final fitted box, truth, and the belief shading.
- `charts/2_belief.png`: how the possible box poses shrink as contacts arrive. Shading is the
  best-fit RMS contact gap if the box centre were at that point.
- `charts/3_estimate.png`: estimated x, y, yaw and the worst contact gap after each contact.
- `charts/4_forces.png`: force residual over every probe, with the 4 N contact and 20 N
  safety lines.
- `charts/5_motion.png`: per path, the duration, off-path joint error and tip lag; stops are
  marked in red.

The localizer is a least-squares box fit to touch contacts, not a particle filter. The belief
charts show the same evidence as a map of where the box can be.

Copy a run to a laptop (PowerShell or any terminal):

```
scp -r student@<lab-pc-ip>:~/16662_RobotAutonomy/src/devel_packages/kilo_project/runs/real/<stamp>-localize .
python scripts/real/view_localization.py <stamp>-localize --show
```

## When something goes wrong

- **`SAFETY STOP: robot reports an active controller error`**: `report.txt` lists the libfranka
  error flags. The next motion clears them; check `./status.sh` first.
- **The robot does not move, or robot_mode is 5**: the e-stop is pressed, or the arm is locked
  in Desk. Release or unlock it, then run `./status.sh`.
- **"another arm client is connected"**: close guide mode (`guide_mode.py`) or the other script.
- **The home check fails**: the straight joint path to home would cross the shelf model. Move
  the arm by hand (guide mode) somewhere more open, then retry.
- **The gripper reads 80 mm after a restart**: the peg was released. Run `./grasp.sh`.
- **To change motion settings**, edit the `joint_executor` section of `configs/real_planned.yaml`:
  - `execution: stream` (default) runs each path as one torque-control skill;
  - `execution: segments` restores the old segment executor;
  - run `./rehearse.sh` after any change.
