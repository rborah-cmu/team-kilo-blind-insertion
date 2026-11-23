"""frankapy backend (the course setup: User PC docker -> Control PC franka-interface at 1 kHz).

Free-space moves use ``goto_pose`` with Franka's internal Cartesian controller.
Probing streams pose + stiffness targets to a dynamic ``goto_pose`` skill under
frankapy's Cartesian impedance controller, following frankapy's
examples/run_dynamic_pose.py. Contact detection runs in Python on
``get_ee_force_torque()`` (libfranka's O_F_ext_hat_K, base frame).

Limits of this backend, measured by the `noise` experiment rather than assumed:
- every state getter is a ROS service call, so the loop rate (target 50 Hz) and
  the latency between contact and stop depend on the lab network;
- frankapy does not expose the per-joint external torques (tau_ext_hat_filtered).

Not yet run on the robot: check each call in the lab with the `check` experiment.
"""

from __future__ import annotations

import time

import numpy as np

from .interface import RobotBackend, RobotState, tool_rotation, yaw_of_tool


class FrankaPyBackend(RobotBackend):
    preferred_hz = 50.0

    def __init__(self, cfg: dict, robot_num: int | None = None):
        import rospy
        from autolab_core import RigidTransform
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy import FrankaArm, SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto import (CartesianImpedanceSensorMessage, PosePositionSensorMessage,
                                    ShouldTerminateSensorMessage)
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg

        self._rospy, self._RigidTransform = rospy, RigidTransform
        self._msg = dict(SensorDataMessageType=SensorDataMessageType, Pose=PosePositionSensorMessage,
                         Impedance=CartesianImpedanceSensorMessage, Terminate=ShouldTerminateSensorMessage,
                         group=make_sensor_group_msg, to_ros=sensor_proto2ros_msg)
        rc, mc = cfg["robot"], cfg["motion"]
        n = int(robot_num or rc.get("robot_num", 1))
        self.fa = FrankaArm(robot_num=n)
        topic = FC.DEFAULT_SENSOR_PUBLISHER_TOPIC if n == 1 else f"franka_ros_interface_{n}/sensor"
        self.pub = rospy.Publisher(topic, SensorDataGroup, queue_size=1000)
        self.probe_length = float(rc["probe_length"])
        self.force_sign = float(rc.get("force_sign", 1.0))
        self.k_trans = [float(k) for k in mc["stream_stiffness"][:3]]
        self.k_rot = [float(k) for k in mc["stream_stiffness"][3:]]
        self.stiffness = min(self.k_trans)
        self.stream_seconds = float(mc.get("stream_skill_seconds", 120.0))
        self.min_move_seconds = float(mc.get("min_move_seconds", 2.0))
        self._id, self._t0, self._streaming, self._rates = 0, 0.0, False, {}

    # ---------------------------------------------------------------- helpers
    def _pose(self, tip, yaw):
        R = tool_rotation(yaw)
        return self._RigidTransform(rotation=R, translation=np.asarray(tip, float) - R[:, 2] * self.probe_length,
                                    from_frame="franka_tool", to_frame="world")

    def _now(self) -> float:
        return self._rospy.Time.now().to_time()

    # ---------------------------------------------------------------- interface
    def state(self) -> RobotState:
        pose = self.fa.get_pose()
        ft = np.asarray(self.fa.get_ee_force_torque(), dtype=float)
        R = np.asarray(pose.rotation)
        return RobotState(time=self._now(), tip=np.asarray(pose.translation) + R[:, 2] * self.probe_length,
                          yaw=yaw_of_tool(R), force=self.force_sign * ft[:3], torque=self.force_sign * ft[3:],
                          q=np.asarray(self.fa.get_joints(), dtype=float))

    def move(self, tip, yaw, speed) -> None:
        current = self.state().tip
        duration = max(self.min_move_seconds, float(np.linalg.norm(np.asarray(tip) - current)) / speed)
        self.fa.goto_pose(self._pose(tip, yaw), duration=duration, use_impedance=False, block=True,
                          ignore_errors=False)

    def begin_stream(self) -> None:
        st = self.state()
        self.fa.goto_pose(self._pose(st.tip, st.yaw), duration=self.stream_seconds, dynamic=True, buffer_time=10,
                          cartesian_impedances=self.k_trans + self.k_rot)
        self._t0, self._id, self._streaming = self._now(), 0, True

    def stream_target(self, tip, yaw) -> None:
        m = self._msg
        self._id += 1
        ts = self._now() - self._t0
        pose = self._pose(tip, yaw)
        traj = m["Pose"](id=self._id, timestamp=ts, position=pose.translation, quaternion=pose.quaternion)
        imp = m["Impedance"](id=self._id, timestamp=ts, translational_stiffnesses=self.k_trans,
                             rotational_stiffnesses=self.k_rot)
        T = m["SensorDataMessageType"]
        self.pub.publish(m["group"](trajectory_generator_sensor_msg=m["to_ros"](traj, T.POSE_POSITION),
                                    feedback_controller_sensor_msg=m["to_ros"](imp, T.CARTESIAN_IMPEDANCE)))

    def tick(self, dt: float) -> None:
        rate = self._rates.get(dt)
        if rate is None:
            rate = self._rates[dt] = self._rospy.Rate(1.0 / dt)
        rate.sleep()

    def end_stream(self) -> None:
        if not self._streaming:
            return
        m = self._msg
        term = m["Terminate"](timestamp=self._now() - self._t0, should_terminate=True)
        self.pub.publish(m["group"](termination_handler_sensor_msg=m["to_ros"](term, m["SensorDataMessageType"].SHOULD_TERMINATE)))
        time.sleep(0.2)
        try:
            self.fa.stop_skill()
        finally:
            self._streaming = False

    def gripper(self, width: float, grasp: bool = False, force: float = 20.0) -> None:
        self.fa.goto_gripper(width, grasp=grasp, force=force, block=True)

    def guide_mode(self, seconds: float) -> None:
        self.fa.run_guide_mode(duration=seconds, block=False)

    def close(self) -> None:
        self.end_stream()
