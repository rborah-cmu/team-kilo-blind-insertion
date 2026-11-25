# Agent handoff: Team Kilo, Phase 1 on the real Franka (2026-11-24)

## CURRENT retry lift succeeded but skill completion timing is inconsistent

- User authorized checked vertical lift to .1072 then full localization retry.
  Run `20261124-171557-localize`: lift completed at
  (.5590436,-.0316958,.1068714), peak residual 1.894 N. Repositioning/descent
  completed, then search reached y=-.09. Three discontinuity recoveries occurred.
  No block contact in this retry, and no fitted pose or verification.
- Final probe path 13 failed at its first segment: expected duration 1.0164 s;
  is_skill_done reported true at .1414 s, while measured tip remained
  (.5647959,-.0901323,.0318557), max endpoint joint error .03780 rad.
  Cached residual at failure .0929 N; current errors and all contact/collision
  flags clear. Historical acceleration discontinuity flag is insufficient to
  diagnose this new failure. No force threshold/endpoint tolerance changes made.
- Critically, termination snapshot was idle with q_d at the prior endpoint;
  error_diagnostics ~.12 s later showed robot_mode=2 and q_d beginning to change.
  Fresh post-run state then showed the arm had reached near the commanded 10 mm
  endpoint. This suggests asynchronous skill start/completion reporting, but
  action status/result and goal IDs were NOT captured, so root cause is unproven.
  Do not just ignore premature completion or enable another retry for this case.
- Read installed frankapy methods (no motion): is_skill_done uses action client's
  wait_for_result and sets fa._in_skill=False on a result; stop_skill only calls
  cancel_goal when _connected and _in_skill. Our adapter stop also gates on
  _in_skill. Review command ownership/cancellation and action result lifecycle
  before another run; an early done report can clear that cancellation gate.
- Fresh post-stop snapshot: idle, no active errors, tip
  (.5547466,-.0900570,.0316294), q
  [-.160839,.449384,-.0000695,-2.171550,-.000181,2.620300,2.195291].
  `runs/real/lab/after-fast-retry-state.json`. Arm is LOW, not home.
- Raw retry log, charts and diagnosis saved locally under
  `runs/real/lab/20261124-171557-localize/`. Retreat completion fix remains staged
  and unchanged. No further hardware motion or new approval pending.

## CURRENT faster localization touched block then stopped on retreat bookkeeping

- User authorized full faster localization. Run `20261124-170644-localize`
  traveled to first corner, descended, and searched through y=-.03. Six pure
  motion-generator discontinuity recoveries occurred (two travel, four probes).
  These remain an unresolved source of delay. All raw logs copied locally to
  `runs/real/lab/20261124-170644-localize/`; charts generated locally.
- Audit path 20 DID detect contact: tip (.5621308,-.0317158,.0316534), peak residual
  5.4834 N; final force residual (-3.6200,-4.0613,.6850) N. User explicitly
  confirmed seeing the peg touch the block. `contacts: []` in the localizer result
  and generated report is misleading: backend retreat failed before probe()
  returned to localizer bookkeeping. Completed contact audit and last_probe
  samples preserve this detection. No fitted block pose or verification yet.
- Path 21 was the 3 mm retreat, whose endpoint was reached (max joint error
  .434 mrad <3 mrad tolerance, no premature end, no active controller errors).
  Existing completion code incorrectly required the 4 N CONTACT debounce to
  clear even for travel. After bias sampling under load, unloading produced an
  11.82 N final residual (<20 N safety limit), so this guard rejected completion.
  This final stop was not a new active controller fault or a missed endpoint.
- Fixed only non-probing completion: completed travel no longer requires
  residual <=contact threshold; both absolute and adaptive 20 N safety checks,
  early-stop refusal, endpoint/posture/geometry checks remain. Probe debounce
  remains unchanged. Successful probe audits now also save force_n/bias_n before
  retreat, so those values survive later failure. 48 executor/experiment tests
  passed, including load release, excessive force, early end and failed retreat.
- Fix staged on lab computer and compiled inside Docker; backup
  `/tmp/kilo-before-retreat-completion-fix-20261124.tar.gz`. Remote hashes match
  local. NO further motion since the stopped run.
- Fresh post-stop snapshot: `runs/real/lab/after-fast-localize-state.json`, idle,
  no current errors; tip (.5590748,-.0317013,.0314833), q
  [-.056547,.445464,-.0000695,-2.179242,-.000184,2.624245,2.299669]. Arm remains
  low near the block, not at home. User was asked to leave arm/block fixed.
- Vertical lift at current xy to z=.1072 passes offline geometry checks
  (2 commands, ~3.79 s commanded). Next concrete motion proposal: lift, then
  rerun full faster localization with up to two verification rounds. Await
  fresh user approval/e-stop coverage; the earlier run's authorization has ended.

## CURRENT hardware result November 24 faster free-air trial passed

- Lab computer IP changed to 172.26.5.186 (student, iam-camilo). SSH key works;
  use HostKeyAlias=172.26.113.205 with the existing known_hosts entry to verify
  the same trusted host. Docker frankapy_docker and Control PC server running.
- Backed up remote config/executor to `/tmp/kilo-before-fast-probes-20261124.tar.gz`
  and locally under `runs/real/lab/pre-fast-update-20261124/`. Remote comparison
  showed only the intended probe_step changes. Staged config and two executor
  files; Docker compile passed, SHA-256 matches local tested files.
- User confirmed e-stop coverage/clear path/peg grip and authorized wrist setup,
  faster 20 mm +x free-air probe and return. Run `20261124-170432-touch` completed
  all four audited paths (startup, small travel correction, probe, return), no
  safety stop/recovery. Logs copied to `runs/real/lab/20261124-170432-touch/`.
- Startup: 19.63 s commanded, max endpoint joint error .303 mrad. Probe:
  19.823 mm traveled, two ~10 mm segments, 2.840 s recorded duration, 74 samples,
  no contact, peak residual 2.904 N (<4 N threshold), no stall. Effective sample
  rate ~25.7 Hz despite configured 20 ms sleep; do not claim measured 50 Hz.
  Probe endpoint joint error <=.413 mrad, wrist yaw error <=.363 mrad.
- Return: 1.100 s commanded, endpoint joint error .689 mrad; tip return error
  .220 mm. Fresh read-only post-check: idle, no current errors; tip
  (0.3072577,-0.0001480,0.4389012), q approximately
  [-.000321,-.784768,-.0000008,-2.356837,-.000191,1.572398,2.355547].
  Snapshot `runs/real/lab/after-fast-free-air-state.json`. Historical motion
  discontinuity flags persist; they were not active errors during this trial.
- Initial localization travel to (.565,-.21,.1072), guarded descent to .0322,
  and full first sweep to (.32,-.21,.0322) checked offline from that snapshot.
  Evidence `fast-localization-start-check.json` in local lab runs folder.
  Subsequent adaptive paths still replan from measured state during execution.
- Full localization has NOT run with faster code. Next: obtain confirmation for
  full search/face probing/up to two verification rounds, with e-stop coverage
  maintained and block fixed. No full-localization motion authorization yet.

## Offline performance update November 24

- User requested faster localization. Added `joint_executor.probe_step` (default
  falls back to dense sampling, real_planned.yaml selects 10 mm). Probe adapter
  merges checked 2 mm IK/collision samples into fewer commands, retaining edge
  certification, full-ray preflight, straightness/orientation tests and continuous
  force polling. Existing motion speeds/caps/thresholds/gains and `path_checks`
  setting are unchanged. Execution remains segments, not streaming.
- Matched full nominal offline localization: both verified; 446.28 -> 293.60 s
  virtual time (34.2% less), 597 -> 167 commands. Peak simulated contact residual
  7.99 -> 10.20 N; hardware force transients remain to be checked. Joint caps
  remain .15 rad/s and .3 rad/s². Representative 245 mm sweep: 68.62 -> 30.16 s;
  face probe: 12.26 -> 5.70 s. No hardware operations for this update.
- 54 executor/experiment/geometry/touch tests passed, including new mid-segment
  contact and fault-stop regressions. Benchmark tool:
  `scripts/real/benchmark_localization_motion.py`; evidence:
  `runs/real/motion-speed/{before.json,after.json,full-before/,full-after/}`.
- Local changes only; no commit/push/deployment. Prior physical-state notes below
  are historical; this update did not reconnect to or inspect the robot.

## Historical November 23 hardware disconnection and incomplete validation

- User confirmed on Nov 23 that the lab computer and arm were disconnected.
  SSH to 172.26.113.205 times out. No motion or remote changes this session.
  Current physical pose/controller state is unknown. After reconnection, read
  state and retrieve the missing log before planning another hardware step;
  obtain fresh e-stop/clear-area confirmation for that concrete motion.
- The authorized wrist setup + 20 mm +x free-air probe/return did NOT pass.
  Trial `20261122-235107-touch` stopped at startup endpoint checking, before
  the probe. Last readback: tip approximately (0.3072,0.0001,0.4389) m,
  q7=2.34562625 vs goal 2.356442756 rad (~10.8 mrad short), idle, no active
  errors. Local log: `runs/real/lab/20261122-235107-touch/`.
- Changed candidate q7 impedance from 50 to 300 (installed factory value 2000)
  and made high-pose startup correct residual orientation errors >4 mrad,
  matching the planner's 5 mrad requirement. No safety bounds were loosened.
  These changes were staged remotely before the second trial.
- Trial `20261122-235722-touch` also reported "joint skill ended early or short".
  Its log has NOT been retrieved. Which stage failed, whether the 20 mm probe
  started, and final pose are UNKNOWN. Retrieve from the Docker project
  `runs/real/20261122-235722-touch` before changing gains or retrying.
- Offline-only diagnostics update now caches command starts/goal/timing and
  feedback, and captures q_d/controller flags on first observed skill completion
  before cleanup resets them. Failed path audits now retain tracking, samples,
  and exception details. Diagnostic query failures do not suppress safety stops.
  This diagnostics update is local only, not staged on the disconnected host.
- Validation for this update: `tests/test_joint_executor.py` plus
  `tests/test_real_experiments.py`: 30 passed. Includes early versus short
  termination, pre-cleanup controller capture, diagnostic-service failure,
  failed-path audit persistence, and JSON serialization of ROS snapshots.
- Earlier "home/yaw zero" and "no new motion" status below is historical and
  superseded. Full localization remains unvalidated on hardware.

## Historical: joint executor implemented, offline tested, staged

- User requested executor changes and offline testing before further hardware.
  Added `kilo/robot/joint_executor.py`, `planned_frankapy.py`, a virtual transport,
  `configs/real_planned.yaml`, and `test_joint_sequence.py`. Factory selects
  planned backend only with `robot.executor: planned_joint`; real.yaml remains
  legacy. Fixed wrist pi / joint 3 zero enforced; no Cartesian fallback.
- Complete rays/edges are planned/collision checked before dispatch. Joint
  commands advance tip <=2 mm, with minimum-jerk speed/accel limits .15 rad/s /
  .3 rad/s^2. Monitor q3, full orientation/yaw, tip tracking, actual keep-outs,
  active errors, feedback freshness and robot mode. Check measured q AND q_d:
  installed JointPositionSkill begins from q_d. Persistent force bias tracking
  across segments; no resume/re-zero after backstops. Early skill completion
  stops even if tiny waypoints fall within joint tolerance. Cleanup stops an
  owned active skill, retries failed stop requests, and preserves failure logs.
- Contact probing permits 6 mrad waypoint lag vs 3 mrad travel, always bounded
  by 4 mm tip / .01 rad q3 / .03 rad orientation checks. Needed to build force
  at corners. A more tangential rotated-face case still safely aborts; this is
  a KNOWN LIMITATION, not a successful localization. No wider limits were used.
- Full test suite: 83 passed before final stale-feedback/grazing regressions;
  final focused `tests/test_joint_executor.py`: 24 passed. Faults cover joint3,
  wrist, tip, force, controller errors, timeout, early end, altered/stale paths,
  Ctrl-C, failed stop, q_d mismatch and stale/user-stopped robot states.
- Docker Python 3.8: compile + synthetic smoke + FULL nominal localization
  through actual joint executor and virtual plant passed. 1486 joint commands,
  825.14 s virtual time, peak joint speed .15, accel .3, max q3 error .0000979
  rad, wrist tracking error ~1.7e-13, min sampled model gap .01068 m. These are
  SOFTWARE TEST-DOUBLE results, not real tracking measurements. Force/corner
  regressions also tested locally; one near-tangential scenario aborts safely.
- `runs/real/joint-executor-release/summary.json` / `docker-replay.json` hold
  final evidence. Executor core hash matches Docker replay. Adapter hardware
  methods are separately tested/compiled. Source staged on User PC; backup
  `/tmp/kilo-before-planned-executor.tar`. No hardware motion during update.
- CURRENT physical arm remains home from prior joint recovery / snapshot.
  Next hardware step: get fresh e-stop confirmation for checked startup wrist
  half-turn at high home, a single 20 mm +x free-air probe and return, using
  `phase1.py touch --config configs/real_planned.yaml --distance 0.02 --repeats 1 --execute`.
  Expect about a minute including slow wrist startup. Hardware tracking is NOT
  validated yet; do not jump straight to full localization. New face touch
  needed after tracking check. No current motion authorization is pending.

The old `execution_ready:false` geometric reports correctly describe the earlier
executor gap. It is now implemented and offline tested, but hardware validation
remains pending. Historical notes below are superseded by this status.

## LATEST: full sequence planned offline; do not run old Cartesian search unchanged

- Last revised hardware search `20261122-224808-localize` descended successfully
  near (0.565,-0.21), then stopped during the first sweep at tip approx
  (0.534,-0.2097,0.0313): inflated RShelfMiddle overlap, no raw shelf overlap,
  no active robot errors, no contacts. Local copy in runs/real/lab/.
- User approved a 10 s joint-space recovery. `20261122-225223-joint-recovery`
  succeeded. All 101 sampled recovery states cleared raw shelves and walls;
  only the initial state touched the inflation margin. Arm is HOME, no errors.
- Latest read-only snapshot `runs/real/lab/planning_state.json`: q approx
  [0.000185,-0.784588,-0.000098,-2.356150,0.000184,1.571203,-0.785232],
  tip (0.307090,0.000077,0.439065), robot idle. NO motion since this snapshot.
- User requested full offline planning before another hardware trial. New
  `kilo/robot/planning.py` matches installed DH/collision geometry exported to
  `configs/frankapy_geometry.json`. Tool-frame -45 degree correction confirmed
  against measured tool and independent MuJoCo kinematics; do NOT rotate the
  collision-link transforms. Prior obstruction is wrist housing box 10/link 7,
  not the elbow. A full tool-pose wrist half-turn removes that housing overlap.
- Planned fixed yaw=pi, joint 3=0. Full adaptive sequence driven by hypothetical
  contacts in `kilo/robot/preflight.py`, through existing TouchLocalizer.
  `scripts/real/preflight_phase1.py` has NO execution mode. Whole commanded probe
  rays checked before contact truncation; actual planned branches include
  startup wrist rotation, travel, descents, probing, retreat, lift, crossings,
  verification and final lift. Edges certify >=2 mm extra clearance beyond
  10 mm shelf inflation using a conservative box-vertex displacement bound.
- `runs/real/preflight-certified/report.json`, `summary.json`, `*-joints.npz`,
  `overview.png`: 10 cases all passed, all 9 box-present cases verified under
  geometric oracle contacts. Cases: nominal, four +/-5 mm / +/-5 degree poses,
  no box (entire snake), +/-25 degree force directions, +/-6 mm search height.
  26,451 joint-state samples passed independent MuJoCo self-collision checking;
  All 11,420 unique states also PASSED the independent installed frankapy check
  in OFFLINE mode: no overlaps, max tip-model difference 7.33e-8 m, 155 s.
  Saved `installed-library-verification.json`; input SHA-256 matches local
  `verification_states.npz`. All report source hashes revalidated unchanged.
  Lowest joint-limit margin 0.08547 rad. Continuous modeled clearance lower
  bound >=0.002002 m beyond the inflated shelves. Supports remain estimates.
- Tests: 62 pass. New geometry/SAT tests include independent vertex-projection
  comparison, frame agreement with MuJoCo, historical failure reproduction,
  half-turn fix, and edge-midpoint collision rejection.
- IMPORTANT: `execution_ready` remains FALSE. Existing Cartesian impedance
  backend cannot guarantee joint-3 posture tracking. Do NOT present an IK path
  as validation of that controller, or simply rerun old `localize`. Need checked
  joint-path execution/posture-aware control plus guarded tracking validation
  before the next full run. No new motion authorization is pending.

Offline planning task completed: full report/summary/preview and joint paths are
in `runs/real/preflight-certified/`. No hardware motion was issued during planning.

Older sections below are historical and may state obsolete robot poses or readiness.

## Latest status: shelf-aware planner ready for another hardware trial

- Full hardware localization `20261122-223313-localize` got two contacts then
  stopped before the third crossing: aligned/perpendicular wrist choices both
  violated the hand wall envelope. No pose estimate was produced. Logs copied
  to `runs/real/lab/20261122-223313-localize/`.
- Trace review found an ignored stalled positioning descent: target z=0.0322,
  actual z=0.0552. Side probing must not continue from such a short descent.
  New checks require search height within 6 mm and >=5 mm finger clearance,
  reject stalled descents, and validate height after wrist rotation/contact.
- User supplied `D:\F26\16662\Lab 2\team_kilo.zip` as shelf reference. All 30
  generated shelf/support boxes EXACTLY match current project simulation XML.
  Extracted into `configs/shelf_obstacles.json` (base-frame metres, full box
  dimensions). Board bounds measured; support shapes estimated in Lab 2 code.
  Left inner y=0.30, right inner y=-0.32. These were absent from the real guard.
  The previous y=0.20 descent with a y-aligned hand could reach into a shelf;
  this is a possible explanation for the stall, not a proven collision.
- Real backend now checks full-arm collision against these boxes inflated by
  10 mm before/during travel and native probing. A read-only check at current
  q found no shelf overlaps; one full 30-box check took ~8.6 ms. frankapy
  REQUIRES NumPy arrays for each box, not Python lists.
- Planar hand guard conservatively respects shelf columns at every height.
  Revised search bounds: x [0.32, 0.565], y [-0.21, 0.15]. Corners clear with
  x-aligned hand. Wrist planner can use a cardinal fallback after aligned and
  perpendicular choices, still checking full swept rotations and joint 7.
- Verification skips faces whose positioning path cannot be planned before
  moving, refits with reachable contacts, and reports `verified`/`ambiguous`.
  Runtime force/height/collision stops still abort. A fit is not verified by
  an all-skipped round. `close()` stops an owned active native skill on errors.
- Offline rehearsal at box (0.477, -0.106, 0): final center
  (0.4771138, -0.1059358), yaw 179.7533 deg, max gap 0.1862 mm,
  verified=True, ambiguous=False; three added verification contacts, one
  unreachable face safely skipped. Full suite 54 pass before final cleanup/
  verification skip additions; latest targeted 28 pass. No new hardware motion
  while preparing this update.
- CURRENT last measured tip: (0.4677, -0.0048, 0.0617), yaw -67.8 deg,
  q=[0.263,0.233,-0.258,-2.404,0.072,2.508,0.344]; robot idle, no active errors.
  Check pose again if user moved it. User now asks to start arm testing again;
  obtain fresh e-stop/readiness confirmation for the revised full localization.

The historical notes below record earlier stages; latest status above takes precedence.

## Follow-up session: wall guard deployed, awaiting manual recovery

### Latest hardware update: slow return home succeeded

#### Subsequent free-air test and return (latest state)

##### Latest: static impedance travel succeeded; preparing descent

###### Latest: guarded descent completed; awaiting single face-touch approval

Latest after the first face-touch attempt:

**LATEST SUCCESS (supersedes pending retry below):**

- User authorized the impedance-controller retry. Run `20261122-223113-touch`
  detected one real face contact at tip approx (0.3978, -0.0759, 0.0371),
  travel 41.43 mm, force approx (-8.0, 0.4, 1.0) N, peak residual 8.05 N,
  direction 177.2 deg (opposes +x). Post-contact retreat and return completed.
  One sample is not a repeatability measurement (spread 0 in summary is trivial).
- Final read-only verification: CURRENT tip (0.3595, -0.0741, 0.0349),
  yaw -0.3 deg, robot mode idle, no active errors. Gripper width 0.0242770 m.
  Log and trace copied locally to `runs/real/lab/20261122-223113-touch/`.
- Native probe controller is now STATIC IMPEDANCE, as is travel. This has now
  performed a hardware face contact. Prior free-air 117 mm validation used the
  internal controller; do not silently generalize its statistics to impedance.
- Next: obtain confirmation for full `localize` (configured 2 verification rounds)
  with a partner at e-stop. No full localization has run in this follow-up yet.

- Authorized `20261122-222834-touch` was refused immediately by the internal
  Cartesian controller; no motion toward the block and no contact. Last raw
  motion error is again joint acceleration discontinuity, with all current
  errors and both collision-flag arrays clear, robot idle.
- CURRENT tip after attempt: (0.3561, -0.0741, 0.0374), yaw -0.2 deg.
- Added `motion.native_probe_controller: impedance` to use static impedance
  for the native guarded probe as well as travel. Contact thresholds, native
  force backstop, detector, and virtual walls stay enabled. This controller's
  contact performance has not been validated yet. 19 targeted tests pass.
- Probe early refusal now includes controller diagnostics; experiment script
  preserves last probe samples on SafetyStop. Await approval to retry ONE
  65 mm +x face probe and return with this controller.

- Authorized vertical native guarded descent completed, run
  `20261122-222703-guarded-descent`: target z=0.0322 at initial xy
  (0.3578986, -0.0725089), no contact, not stalled, travelled 0.4004243 m,
  peak detector residual 3.19684 N. Final tip (0.3558791, -0.0738109,
  0.0373055): 5.1 mm high from tracking error. Still valid shaft/box overlap
  with ample finger clearance. Robot idle, current errors empty.
- Guarded descent did not itself perform a box contact. Next user confirmation
  requested for one +x `touch`, distance 0.065 m, repeats=1, at current height,
  then return using static impedance. Expected side contact after about 45 mm
  if the robot-facing face remains at x~0.414. No full localize yet.
- Travel update deployed with Docker compile and stationary `check` successful.
  Previous deployed source/config backup: `/tmp/kilo-before-travel-fix-20261122.tar`;
  update archive: `/tmp/kilo-phase1-travel-fix.tar`.

- With user confirmation, static Cartesian impedance `goto_pose` to tip
  (0.360, -0.075, 0.439), duration 6 s, gains [1500,1500,1500,100,100,100],
  raw per-axis force backstop |idle force| + 20 N and torque backstop
  |idle torque| + 10 Nm, virtual walls enabled, SUCCEEDED.
- Run `20261122-222350-impedance-travel`: actual tip
  (0.3579, -0.0725, 0.4378), target error 0.00343567 m. Robot idle, current
  errors empty; historical last-motion error still records earlier Cartesian
  acceleration discontinuity. CURRENT arm pose is this high near-side pose.
- Local changes select static impedance for free-space `move` (native guarded
  probes remain internal Cartesian), use 20 mm/s travel with 6 s minimum,
  same gains/backstop as successful trial, and use that travel primitive for
  post-contact retreats. Failure messages now expose current vs historical
  robot errors. `touch` saves each probe before trying its return.
- Targeted tests: 17 passed. No descent/contact commanded yet. Next request
  confirmation for guarded descent beside block to z=0.0322 at x approx 0.36,
  y approx -0.075; then single +x box-face touch; then full localization.

- User explicitly authorized the 12 cm +x free-air `touch` test and return.
  Run `20261122-221932-touch` travelled 117.1 mm at 10 mm/s, no contact,
  peak residual 2.1389 N (threshold 4 N), 99th percentile 2.0525 N.
  Saved probe trace has 290 samples over 12.1308 s. Logs copied locally to
  `runs/real/lab/20261122-221932-touch/`.
- Automatic Cartesian return at 50 mm/s stopped immediately, 117 mm short.
  One return retry at 10 mm/s also stopped immediately. Raw ROS last error
  is `cartesian_motion_generator_joint_acceleration_discontinuity`; current
  errors are empty and robot idle. Slowing alone did not resolve this.
  Do not claim the return succeeded from the detector's successful forward leg.
- With fresh explicit user approval, a 10-second joint-space return SUCCEEDED
  using [0, -0.785, 0, -2.356, 0, 1.571, -0.785] (q7 preserves x-aligned wrist).
  All 41 interpolated joint samples passed Python wall checks before motion;
  virtual walls remained enabled. Maximum joint error 0.0008122 rad.
- CURRENT tip: (0.3073245, -0.0000638, 0.4391937) m; finger yaw approximately
  zero; robot idle, no current errors. No box-face touch or descent has run.
- Next: validate another free-space travel strategy (e.g. a slow static
  Cartesian impedance move) before descending toward the box. Native guarded
  forward probing and internal joint-space moves worked; internal Cartesian
  return moves have failed at both speeds. Root cause is not established.
  Ask per motion step, as before. Also improve `cmd_touch` to preserve the
  per-probe result before its return move: current log has samples but loses
  the structured contact summary if the return raises SafetyStop.

- User recovered by hand to tip (0.4901, -0.0509, 0.1793), yaw -1.1 deg.
  The first authorized positioning attempt (`20261122-221434-move`) was refused
  before descent, ending 132 mm short of the lateral target. Raw ROS state
  identifies `last_motion_errors.cartesian_motion_generator_joint_acceleration_discontinuity`.
  Current errors were all false, robot mode idle, and Python full-arm wall
  collision flags all false. The failure message's wall/limit suggestion was
  not a diagnosis. Separate gripper getter returned 0.0242776 m (the raw arm
  state gripper fields are zero and must not be used as gripper truth).
- After explicit user confirmation, executed `goto_joints` to configured home
  [0, -0.785, 0, -2.356, 0, 1.571, 0.785], duration 10 s, internal joint
  controller, virtual walls enabled. Rechecked 41 interpolated configurations
  against frankapy wall boxes before execution; all clear.
- Home succeeded: maximum joint target error 0.0008361 rad; tool position
  (0.3072049, -0.0001462, 0.4864945), peg tip approximately
  (0.3071902, -0.0001192, 0.4388695), finger yaw approximately -90 deg.
  Robot mode idle, all current errors false. Last-motion error still retains
  the earlier Cartesian acceleration error; it is historical, not active.
- Pending user confirmation for one free-air `touch` experiment at this height:
  align wrist along x, +x 0.12 m at 0.01 m/s, repeats=1, then return to start.
  No descent or box-face touch has run in this follow-up session.

This section supersedes the unfinished wall-guard work below. No robot motion
had been commanded at the time of the original follow-up notes below; see the
latest hardware update above for subsequent motion.

- Added `kilo/robot/workspace.py`: configurable conservative planar hand envelope,
  checking translation and the complete shortest wrist rotation. The original
  frankapy virtual walls remain enabled; their constants were read from Docker
  and agree with the planar bounds recorded below.
- Localizer chooses aligned/perpendicular wrist orientation, checks crossing
  start AND end, rotates before travel, and selects the first travel yaw for
  the upcoming snake sweep. Backend also guards direct moves/probes/noise.
- Fixed native `probe` returning an uninitialized baseline when already within
  2 mm of its target. Added deterministic native force-trace tests for 12 cm
  free-air drift, blocked-tip contact, force safety stop, and near-target return.
- Final full suite passed 43 tests, including the first-sweep regression test;
  the targeted suite passed 15 tests. Docker compilation passed under
  Python 3.8, but Docker has no pytest installed.
- A fixed-pose MuJoCo rehearsal at box (0.477, -0.075, 0) completed 4 initial +
  4 verification touches, estimated centre (0.4770235, -0.0749993), yaw 179.9594
  degrees, max gap 0.0257 mm. Random seed 8 was safely refused during a crossing
  near the wall: do not assume every arbitrary box pose fits this workspace.
- Synced the five existing changed source/config files plus new workspace module,
  two new tests, and updated lab guide to the User PC. Previous deployed files:
  `/tmp/kilo-before-wall-fix-20261122.tar`; deployment archive:
  `/tmp/kilo-phase1-wall-fix.tar`. New files must be included in future syncs.
- SSH now works with the user's RSA key. In the sandbox, explicit options were
  needed: `-o UserKnownHostsFile=C:/Users/rianb/.ssh/known_hosts
  -i C:/Users/rianb/.ssh/id_rsa` (with approved execution outside the sandbox).
- Read-only check `20261122-220701-check`: tip (0.5871, -0.1999, 0.1061), yaw
  -0.6 degrees, 49.6 Hz, force std (0.019, 0.026, 0.028) N. No competing ROS
  `franka_arm_client` was present before this check. Pose still matches wall stop.
- User initially confirmed recovery and e-stop coverage, but measured pose is
  still outside the new guard. Asked them to manually recover to x=0.40-0.45,
  y approximately -0.075, z >= 0.11 and restart the control server, then confirm
  e-stop coverage. Recheck pose before any motion. No virtual-wall bypass allowed.

Next: verify recovered pose, obtain confirmation for each concrete motion step,
then test box-face contact and a 12 cm free-air single `touch` probe. `noise`
records force but does not exercise the native contact detector. Only proceed
to full localization after those detector checks succeed on hardware.

Read this first if you are an AI agent picking up this project. It records the
current state, everything learned on the real robot, and the next steps.

## Project in one paragraph

16-662 Robot Autonomy group project (Team Kilo). A Franka Panda grasps a peg
(25 mm square top, 25 mm round shaft with a 1 mm chamfer, 38.1 mm shaft length)
from a known holder, finds a taped-down box (126.7 x 169 x 50.8 mm, 25.4 mm
through-hole ~4 mm off-centre, 0.4 mm clearance) by touch using only joint
sensing, then inserts the peg (Phase 2, RL residual policy, done by a teammate in
simulation). This session built and is testing **Phase 1 on the real robot**.
Deadlines: progress talk Nov 30 / Dec 2; report (2-3 pages IEEE) and video Dec 11.

Key docs: `README.md`, `docs/real_robot.md` (lab guide), `proposal/proposal.tex`,
`configs/real.yaml` (real-robot settings, with measured values and comments).

## Repo and git state

- Local repo: `D:\F26\16662\Project`, branch `phase1-real-robot` (not pushed).
  Last commit `536f880`. GitHub: `rborah-cmu/team-kilo-blind-insertion` (private);
  `main` there has newer teammate commits.
- **Uncommitted** (all from today's lab session): `configs/real.yaml`,
  `kilo/localization/touch_localizer.py`, `kilo/robot/frankapy_backend.py`,
  `kilo/robot/interface.py`, `scripts/real/phase1.py`. Local lab logs are in
  `runs/real/lab/` (gitignored).
- Python: project `.venv` (`.venv\Scripts\python.exe`): torch 2.14.1 CPU, sb3 2.9.2,
  gymnasium 1.2.3 (1.4.0 fails on Python 3.10.5), mujoco 3.17. Global Python keeps
  torch 2.5.1 for the user's torchvision; do not install into it.
- Tests: `.venv\Scripts\python.exe -m pytest tests` (28 pass).

## Lab setup (how to reach the robot)

- User PC: `student@172.26.113.205` (hostname `iam-camilo`). SSH works with the
  local key (`ssh -o BatchMode=yes ...`). **Never use a password.**
- Control PC: `iam-sneezy` (frankapy server runs there; started from the User PC with
  `cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-sneezy`).
- Docker `frankapy_docker` on the User PC (Ubuntu 20.04, ROS Noetic, Python 3.8.10).
  Mounts `~/16662_RobotAutonomy/src/devel_packages` -> `/home/ros_ws/src/devel_packages`.
- Code on the lab PC: `~/16662_RobotAutonomy/src/devel_packages/kilo_project`
  (copied from commit 536f880, then individual files synced with `scp`). Keep it in
  sync: `scp <file> student@172.26.113.205:16662_RobotAutonomy/src/devel_packages/kilo_project/<file>`.
- Run a command in the docker (ROS must be sourced):
  ```
  ssh -o BatchMode=yes student@172.26.113.205 'docker exec frankapy_docker bash -c "source /opt/ros/noetic/setup.bash; source /home/ros_ws/devel/setup.bash; cd /home/ros_ws/src/devel_packages/kilo_project && python3 scripts/real/phase1.py check"'
  ```
- Motion commands need `--execute` and type `RUN` at a prompt; over SSH pipe it:
  `printf "RUN\n" | ssh ... 'docker exec -i frankapy_docker ...'` (one RUN per prompt;
  `--start` adds a prompt). For long runs (localize), launch detached with
  `nohup setsid ... > /tmp/kilo_localize.log` and wait with a loop on
  `ps -C python3 -o args= | grep -c scripts/real/phase1` (a plain `pgrep -f phase1`
  matches its own command line).
- Before any motion: confirm with the user that someone is at the e-stop and the area
  is clear. Ask per motion step. Never bypass virtual walls without explicit consent.
- Check nobody else drives the robot: `rosnode list` must not show `franka_arm_client`.

## Experiment script

`scripts/real/phase1.py <command>`: `check` (no motion), `move --start X Y Z`,
`grasp --stage open|close`, `table`, `noise`, `touch`, `groundtruth`, `localize`,
`refit`. Logs go to `runs/real/<time>-<command>/` on whichever machine runs it.
`--backend sim` rehearses offline with MuJoCo (sensor bias/noise added).

## What we learned on the real robot (all measured 2026-11-22)

| Finding | Value / consequence |
| --- | --- |
| State loop rate | ~50 Hz with 3 frankapy service calls per tick |
| Force estimate at rest | offset ~2 N, noise ~0.01-0.07 N std |
| `get_ee_force_torque()` sign | reports the force the robot applies: `robot.force_sign: -1` |
| Table height (peg tip) | `table_z: 0.0072` (4 descents at (0.405,-0.075): 7.1/7.4/7.2/7.0 mm) |
| Fingers vs box top | finger lower edge is 38.1 mm above the tip; at 15 mm tip height they cleared the box by 2.3 mm and once landed on it. Now `search_offset: 0.025` and the localizer refuses < 5 mm clearance |
| frankapy dynamic skills | end after `buffer_time` (not `duration`); streamed targets also stopped being tracked (~23 mm then stall). Use `frankapy_probe_mode: native` |
| Native probe | `goto_pose(block=False)` + Python detection + `fa.stop_skill()`; frankapy `force_thresholds` only as a backstop (idle offset + 20 N), resume if it trips on drift |
| Controller-start jump | force estimate shifts ~2.5 N within 0.25 s of a skill starting, before motion |
| **Drift while moving** | estimate drifts ~0.2 N per mm travelled with no contact -> caused a false contact. Fix: baseline follows force in proportion to tip travel (5 mm); contact = jump of 4 N over it held 3 samples. Tested only against a fake arm so far |
| Tracking error | at 600 N/m the arm settles ~6.5 mm short; now 1500 N/m for probes |
| Peak contact force | 11-15 N on table/box contacts (backstop/holding); safety stop 20 N |
| Touch repeatability | box face contacts within 1.1 mm (4 of 5); first contact 3.5 mm early (box settling or slip) |
| Peg rotates in grasp | ~2 mm at the tip with a fixed wrist -> `align_wrist: true` (finger axis along motion) |
| Joint 7 | dq7 ~ -d(yaw) + d(azimuth) (measured in sim). `choose_wrist_yaw` picks yaw or yaw+pi to keep q7 near 0 |
| **Virtual walls** | frankapy `WORKSPACE_WALLS`: x in [-0.405, 0.745], y in [-0.455, 0.455] (inner faces), floor z -0.05. Checked against the arm's collision boxes during every skill. If the hand touches a wall, EVERY later motion (incl. `reset_joints`) is refused instantly with no robot error. Recovery so far: e-stop, hand-guide, release, restart the Control PC server |
| Hand footprint (from wall stops) | ~0.16 m along the finger axis, ~0.09 m across it, from the tip axis |
| Search area | reduced to x 0.32-0.62, y +/-0.20 (`workspace.search_corners`); still too far for a hand aligned along x near x = 0.62 |

## Current robot state (end of session)

- Arm stopped at the x wall: peg tip ~(0.587, -0.200, 0.106), wrist yaw ~0 deg
  (finger axis along x). **Assume all motions are refused**; ask the user to recover
  by hand (e-stop -> squeeze wrist buttons -> move back/up -> release -> restart
  Control PC server), or with explicit consent one `goto_joints`/`reset_joints` with
  `ignore_virtual_walls=True`.
- Peg is in the gripper (24.3 mm closed width). Box taped with its robot-facing face at
  x ~0.414 m (centre roughly (0.477, -0.075) if square).
- The last localize attempt (`20261122-215541-localize` on the lab PC) stopped on
  the first travel move (33 mm short) because of the wall. The one before
  (`214907`) got a false first contact from drift.

## Next steps (in order)

1. **Finish wall-aware wrist choice** (was in progress, not yet written): in
   `TouchLocalizer._yaw_for`, after `choose_wrist_yaw`, check the hand footprint
   (0.165 m along the finger axis, 0.095 m across, + 0.01 margin) against the walls
   above at both the probe start and end (and travel destinations). If the aligned yaw
   violates, use yaw + 90 deg; if both violate, raise `SafetyStop` before moving. Put
   wall limits and footprint in `configs/real.yaml`. Also make `travel()`'s first
   move use a wall-safe yaw (it currently keeps the current yaw).
2. Optionally add a planner pre-flight that checks every snake waypoint and crossing
   start for wall clearance, and shrink `search_corners` if needed.
3. Unit-test the footprint check; run `pytest`; sync the 5 changed files plus
   `configs/real.yaml` to the lab PC.
4. Get the arm off the wall (user), verify with `check`, then validate the new
   drift-tracking probe on the robot: `touch` at the box face (expect contacts within
   ~1 mm, no false contacts in the free-air run-up), then a free-air `noise`-length
   probe to confirm no false contact over 10+ cm.
5. Rerun `localize` (detached). Then `groundtruth` (hand-guide the peg into the hole)
   and score `localize --truth runs/real/groundtruth.json`; repeat with 5-10 box poses.
6. Check peg slip (pen mark on the shaft); raise `gripper.grasp_force` if it moves.
7. Commit the session's changes on `phase1-real-robot` (ask the user first), then
   push/PR when they agree. Update `docs/real_robot.md` with the lab findings above.
8. Hole search is still TBD (Phase 2's policy handles <= ~5 mm start error; the CAD
   hole is 4 mm off-centre, so the box-centre estimate may suffice; see
   `docs/experiments/end_to_end.json`: 43/43 inserted when Phase 1 was accurate).
