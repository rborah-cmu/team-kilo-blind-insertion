# Proprioceptive peg insertion progress and results

Status as of November 24, 2026

The project has demonstrated guarded contact with the block on the real Franka Panda and developed a shelf-aware planner and joint-path executor for Phase 1 localization. Full block localization has passed offline tests, but has not completed successfully on the real arm. The immediate remaining task is to validate the new executor's joint tracking before repeating contact and localization experiments.

## Project objective and current scope

The overall objective is peg insertion using proprioceptive feedback. Phase 1 estimates the block's planar position and orientation from robot motion, contact locations, and estimated external forces. The sequence searches for the block, probes additional faces, fits its pose, and collects verification contacts. Work so far has concentrated on making this sequence executable within the real shelf workspace. Hardware insertion results are not yet available.

## Real robot results

Hardware experiments on November 22 established the following results. Positions refer to the estimated peg tip in the robot base frame.

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Free-air probe, commanded 120 mm in +x | Approximately 117.1 mm traveled; peak detector residual 2.14 N, below the 4 N contact threshold | The forward probe completed without a false contact. Its Cartesian return failed; a separately authorized joint-space return succeeded. |
| Static Cartesian impedance travel | Reached the high pose near (0.360, -0.075, 0.439) m with approximately 3.44 mm position error | Demonstrated a usable travel primitive for the earlier controller configuration. |
| Guarded vertical descent | Approximately 400.4 mm traveled; peak residual 3.20 N; ended approximately 5.1 mm above the requested height | Descent completed without detected contact, but showed tracking error that subsequent height checks must account for. |
| Single guarded face probe with impedance control | Detected contact after 41.43 mm at approximately (0.3978, -0.0759, 0.0371) m; peak residual 8.05 N; retreat and return completed | Demonstrated real block contact detection. Force direction was 177.2 degrees, opposing the +x probe. Only one trial was collected, so repeatability is unmeasured. |
| Full localization attempt | Recorded two contacts, then stopped when no permitted wrist orientation was available for the next path | No completed pose estimate. Review also identified an inadequately checked positioning descent; the two contacts do not establish successful localization. |
| Revised localization with shelf checks | Descended, then stopped during the first sweep on an inflated shelf keep-out; no contacts recorded | The modeled safety margin was entered. Review found no overlap with the raw shelf boxes at the recorded stopping pose. |
| New joint executor startup and 20 mm free-air trial | First trial stopped during wrist startup, approximately 10.8 mrad short of the joint 7 target; the probe did not run | Joint tracking remains unvalidated. A second trial also stopped early or short; its log must be retrieved to identify the failed stage. |

Earlier internal Cartesian moves reported acceleration discontinuity errors, including return failures that persisted after reducing speed. Joint-space recovery moves succeeded. These observations motivated explicit joint-path execution, but do not establish a single cause for every earlier failure.

## Planning and control progress

The workspace model now includes 30 shelf and support boxes from the previous lab, with a 10 mm inflation margin. Shelf board bounds were measured in that reference; support geometry remains estimated. The planning model also includes the robot's collision geometry, virtual walls, table, and joint limits. A fixed 45 degree tool-frame correction brought the planning kinematics into agreement with measured and independently computed tool poses.

Offline analysis reproduced the shelf interference at the wrist housing. A fixed wrist yaw of 180 degrees with joint 3 near zero clears the modeled probing sequence. The new executor plans complete probe rays before motion and dispatches checked joint segments with tip advances of at most 2 mm. It limits peak joint speed to 0.15 rad/s and acceleration to 0.3 rad/s², and monitors posture, tip tracking, modeled clearance, forces, controller errors, and feedback freshness.

Travel endpoint tolerance is 3 mrad; guarded probing permits 6 mrad. Both retain the 4 mm tip tracking, 10 mrad joint 3, and 30 mrad orientation bounds. Early controller termination prevents further dispatch. Force detection uses a persistent probe baseline, a 4 N contact threshold, and a 20 N safety limit, with controller backstops and virtual walls retained.

After the first startup failure, candidate joint 7 stiffness increased from 50 to 300, below the installed default of 2000. Startup now corrects small orientation residuals at a high pose. The subsequent failure still requires diagnosis. The latest local update preserves termination timing, measured and desired joints, controller flags, and failed-path tracking data before cleanup resets that information; it has not been staged on the disconnected lab computer.

## Offline validation results

| Validation | Result | Scope |
| --- | --- | --- |
| Complete sequence geometry checks | All 10 scenarios passed: nominal pose, four position and yaw perturbations, no block, two force-direction perturbations, and two probing-height offsets | Uses hypothetical contacts; all nine block-present cases verified a pose. |
| Independent installed frankapy geometry check | All 11,420 unique states passed; maximum tip-model discrepancy approximately 7.33e-8 m | Confirms agreement with the installed geometry implementation. |
| Independent MuJoCo self-collision check | 26,451 sampled states passed | Checks sampled configurations; continuous shelf clearance is handled separately by the planner. |
| Continuous modeled path clearance | Minimum certified lower bound approximately 2.002 mm beyond the inflated shelf geometry | A modeled clearance guarantee, subject to geometry and calibration accuracy. |
| Nominal localization through the actual executor in Docker | Passed using a virtual joint/contact plant: 1,486 commands and approximately 825 seconds of virtual time | Software integration result; does not measure real tracking or contact dynamics. |
| Latest diagnostics and experiment regression tests | 30 passed | Includes early termination, endpoint lag, pre-cleanup snapshots, diagnostic failures, failed-path logs, and existing executor safety cases. |

A nearly tangential contact with a rotated block still causes the virtual executor to stop at bounded tracking lag before establishing the contact threshold. Robust localization across all tested contact geometries remains unfinished. Offline fit errors and virtual tracking precision should not be reported as real robot localization accuracy.

## Remaining experiments

The user confirmed on November 23 that the lab computer and arm were disconnected. The arm's current pose is unknown. The next sequence is:

1. Reconnect, retrieve the missing second-trial log, and inspect fresh robot/controller state.
2. Stage the diagnostic update and resolve the observed startup termination using the recorded data.
3. With fresh e-stop coverage and motion authorization, complete the checked wrist setup and a 20 mm free-air probe with return.
4. Repeat guarded face touches to measure contact reliability, location spread, force peaks, and return consistency.
5. Complete localization across several known block placements and compare estimated position and yaw with independently measured ground truth.
6. Report localization error, completion rate, verification residuals, runtime, and safety stops before progressing to insertion trials.

## Supporting records

- [Successful hardware face probe](D:/F26/16662/Project/runs/real/lab/20261122-223113-touch/result.json)
- [Hardware localization with two contacts](D:/F26/16662/Project/runs/real/lab/20261122-223313-localize/result.json)
- [Shelf-aware hardware localization stop](D:/F26/16662/Project/runs/real/lab/20261122-224808-localize/result.json)
- [Offline geometry report](D:/F26/16662/Project/runs/real/preflight-certified/report.json)
- [Offline executor release summary](D:/F26/16662/Project/runs/real/joint-executor-release/summary.json)
- [Detailed handoff and experiment history](D:/F26/16662/Project/docs/AGENT_HANDOFF.md)
