#!/usr/bin/env python3
"""Capture an SO101 follower joint pose as a JSON start pose.

Two modes are supported:
  1. Manual: move the follower by hand, then press ENTER.
  2. Leader teleop: move the leader arm while the follower mirrors it, then
     press ENTER to save the follower pose.

The output JSON can be passed to collection/record_with_start_pose.py.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras import CameraConfig  # noqa: F401 - registers camera config choices
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.common.control_utils import init_keyboard_listener, is_headless
from lerobot.robots import (  # noqa: F401 - imports register robot config choices
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401 - imports register teleop config choices
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class CaptureStartPoseConfig:
    robot: RobotConfig
    # Optional leader arm used to drive the follower to the desired pose.
    teleop: TeleoperatorConfig | None = None
    output: Path = Path("start_pose.json")
    fps: int = 10
    # Maximum time to keep the leader-control loop alive before auto-capturing.
    control_time_s: float = 120.0
    disable_torque_for_manual_positioning: bool = True
    prompt_before_read: bool = True
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False


def _disable_torque_if_available(robot) -> None:
    """Disable motor torque on supported robots so the arm can be moved by hand."""
    bus = getattr(robot, "bus", None)
    if bus is None or not hasattr(bus, "disable_torque"):
        print("Torque disable not available for this robot type; leaving torque state unchanged.")
        return
    bus.disable_torque()
    print("Motor torque disabled. Move the follower arm by hand to the desired start pose.")


def _extract_motor_pose(obs: dict) -> dict[str, float]:
    pose = {key: float(value) for key, value in obs.items() if key.endswith(".pos")}
    if not pose:
        raise RuntimeError(
            "No '*.pos' motor position keys were found in the robot observation. "
            f"Observation keys: {sorted(obs)}"
        )
    return pose


def _save_pose(output: Path, pose: dict[str, float]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pose, indent=2) + "\n")
    print(f"Saved start pose to: {output}")
    print(json.dumps(pose, indent=2))


def _start_enter_listener(done: threading.Event) -> threading.Thread:
    def wait_for_enter() -> None:
        input("Move the leader to the desired start pose, then press ENTER to save it...\n")
        done.set()

    thread = threading.Thread(target=wait_for_enter, daemon=True)
    thread.start()
    return thread


def _leader_capture_loop(robot, teleop, cfg: CaptureStartPoseConfig) -> dict[str, float]:
    if cfg.display_data:
        init_rerun(session_name="capture_start_pose", ip=cfg.display_ip, port=cfg.display_port)

    listener, events = init_keyboard_listener()
    done = threading.Event()
    _start_enter_listener(done)

    control_interval = 1.0 / cfg.fps
    deadline = time.perf_counter() + cfg.control_time_s
    last_obs = None

    print("Leader control active.")
    print("  - Move the leader arm until the follower is in the desired start pose.")
    print("  - Press ENTER in this terminal to save the follower pose.")
    print("  - Right arrow also exits and saves; ESC exits without saving.")

    try:
        while not done.is_set() and time.perf_counter() < deadline:
            loop_t = time.perf_counter()

            if events["stop_recording"]:
                raise KeyboardInterrupt("ESC pressed; capture aborted.")
            if events["exit_early"]:
                break

            obs = robot.get_observation()
            act = teleop.get_action()
            robot.send_action(act)
            last_obs = obs

            if cfg.display_data:
                log_rerun_data(
                    observation=obs,
                    action=act,
                    compress_images=cfg.display_compressed_images,
                )

            precise_sleep(max(control_interval - (time.perf_counter() - loop_t), 0.0))

        if last_obs is None:
            last_obs = robot.get_observation()

        return _extract_motor_pose(last_obs)
    finally:
        if not is_headless() and listener:
            listener.stop()


@parser.wrap()
def capture_start_pose(cfg: CaptureStartPoseConfig) -> None:
    init_logging()
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    try:
        robot.connect()

        if teleop is not None:
            teleop.connect()
            pose = _leader_capture_loop(robot, teleop, cfg)
            _save_pose(cfg.output, pose)
            return

        if cfg.display_data:
            init_rerun(session_name="capture_start_pose", ip=cfg.display_ip, port=cfg.display_port)
            obs = robot.get_observation()
            log_rerun_data(
                observation=obs,
                action=None,
                compress_images=cfg.display_compressed_images,
            )

        if cfg.disable_torque_for_manual_positioning:
            _disable_torque_if_available(robot)

        if cfg.prompt_before_read:
            input("Place the robot in the desired start pose, then press ENTER to capture it...")

        pose = _extract_motor_pose(robot.get_observation())
        _save_pose(cfg.output, pose)

    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()


def main() -> None:
    register_third_party_plugins()
    capture_start_pose()


if __name__ == "__main__":
    main()
