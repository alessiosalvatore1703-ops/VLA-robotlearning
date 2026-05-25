#!/usr/bin/env python3
"""Run a remote MolmoAct2 policy server on a local SO100/SO101 robot.

The robot and camera stay connected to this local machine. Each control cycle:
  1. reads the local LeRobot observation,
  2. sends image + state + task to a remote /act server,
  3. sends returned absolute joint-position targets to robot.send_action.

The remote server decides which model is used. Check:
  curl http://127.0.0.1:8000/health
before running this script.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.cameras import CameraConfig  # noqa: F401 - registers camera config choices
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
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
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from query_molmoact2_once import _image_to_base64_png, _post_json, _select_image


@dataclass
class MolmoAct2RemotePolicyConfig:
    robot: RobotConfig
    task: str = "Place the coke on Taylor Swift."
    server_url: str = "http://127.0.0.1:8000/act"
    camera_key: str | None = None
    num_steps: int = 10
    timeout_s: float = 120.0
    duration_s: float = 20.0
    fps: int = 10
    actions_per_query: int = 3
    # Max absolute change sent per joint per control tick. Set <=0 to disable.
    max_delta_per_step: float = 5.0
    require_action_dim: int = 6
    save_last_image_path: Path | None = Path("molmoact2_last_rollout_image.jpg")
    save_last_response_path: Path | None = Path("molmoact2_last_rollout_response.json")


def _make_observation_payload(robot, features: dict, cfg: MolmoAct2RemotePolicyConfig):
    raw_obs = robot.get_observation()
    frame = build_dataset_frame(features, raw_obs, prefix=OBS_STR)

    if "observation.state" not in frame:
        raise RuntimeError(f"No observation.state found. Frame keys: {sorted(frame)}")

    state = np.asarray(frame["observation.state"], dtype=np.float32)
    image_key, image = _select_image(frame, cfg.camera_key)

    if cfg.save_last_image_path is not None:
        cfg.save_last_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cfg.save_last_image_path)

    payload = {
        "task": cfg.task,
        "state": state.tolist(),
        "images": [_image_to_base64_png(image)],
        "num_steps": cfg.num_steps,
    }
    return payload, state, image_key


def _parse_actions(result: dict, require_dim: int) -> np.ndarray:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise RuntimeError(f"Expected actions with shape (T, D), got {actions.shape}.")
    if actions.shape[1] != require_dim:
        raise RuntimeError(f"Expected action dimension {require_dim}, got shape {actions.shape}.")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Remote policy returned non-finite actions.")
    return actions


def _state_from_robot(robot, features: dict) -> np.ndarray:
    raw_obs = robot.get_observation()
    frame = build_dataset_frame(features, raw_obs, prefix=OBS_STR)
    return np.asarray(frame["observation.state"], dtype=np.float32)


def _clip_target(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return target
    return np.clip(target, current - max_delta, current + max_delta)


def _send_action_vector(robot, action_keys: list[str], target: np.ndarray) -> None:
    action = {key: float(value) for key, value in zip(action_keys, target, strict=True)}
    robot.send_action(action)


@parser.wrap()
def run_molmoact2_remote_policy(cfg: MolmoAct2RemotePolicyConfig) -> None:
    init_logging()

    robot = make_robot_from_config(cfg.robot)
    control_interval_s = 1.0 / cfg.fps

    try:
        print("Connecting robot and camera...")
        robot.connect()

        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
        action_keys = list(robot.action_features.keys())
        if len(action_keys) != cfg.require_action_dim:
            raise RuntimeError(
                f"Expected {cfg.require_action_dim} action keys, got {len(action_keys)}: {action_keys}"
            )

        print(f"Task: {cfg.task}")
        print(f"Action keys: {action_keys}")
        print(f"Server URL: {cfg.server_url}")
        print(f"Duration: {cfg.duration_s}s, fps: {cfg.fps}, actions_per_query: {cfg.actions_per_query}")
        print(f"Max delta per step: {cfg.max_delta_per_step}")
        print("Starting remote MolmoAct2 rollout. Press Ctrl+C to stop.")

        deadline = time.perf_counter() + cfg.duration_s
        query_idx = 0
        sent_count = 0

        while time.perf_counter() < deadline:
            payload, state, image_key = _make_observation_payload(robot, obs_features, cfg)
            print(f"\nQuery {query_idx}: state={state.tolist()} image={image_key}")

            result = _post_json(cfg.server_url, payload, cfg.timeout_s)
            if cfg.save_last_response_path is not None:
                cfg.save_last_response_path.parent.mkdir(parents=True, exist_ok=True)
                cfg.save_last_response_path.write_text(json.dumps(result, indent=2) + "\n")

            if "policy_path" in result:
                print(f"Remote policy: {result['policy_path']}")

            actions = _parse_actions(result, cfg.require_action_dim)
            horizon = min(cfg.actions_per_query, len(actions))
            print(f"Received actions: {actions.shape}; executing first {horizon}")

            for predicted in actions[:horizon]:
                step_start = time.perf_counter()
                current = _state_from_robot(robot, obs_features)
                target = _clip_target(predicted, current, cfg.max_delta_per_step)
                print(
                    f"  send {sent_count}: predicted={predicted.tolist()} "
                    f"current={current.tolist()} target={target.tolist()}"
                )
                _send_action_vector(robot, action_keys, target)
                sent_count += 1

                elapsed = time.perf_counter() - step_start
                precise_sleep(max(control_interval_s - elapsed, 0.0))

                if time.perf_counter() >= deadline:
                    break

            query_idx += 1

        print(f"Rollout finished. Sent {sent_count} actions.")

    finally:
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    try:
        run_molmoact2_remote_policy()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
