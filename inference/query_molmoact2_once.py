#!/usr/bin/env python3
"""Capture one SO100/SO101 observation and query a MolmoAct2 server once.

This script is for the first zero-shot safety test:
  - connect to the robot and camera through LeRobot,
  - read one observation,
  - send image + observation.state + prompt to a remote MolmoAct2 server,
  - print the predicted action chunk,
  - disconnect without sending any action to the robot.

The server should be reachable through an SSH tunnel at http://127.0.0.1:8000.
"""

import base64
import io
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
from lerobot.utils.utils import init_logging


@dataclass
class QueryMolmoAct2OnceConfig:
    robot: RobotConfig
    task: str = "Put the banana in the red colored bowl."
    server_url: str = "http://127.0.0.1:8000/act"
    camera_key: str | None = None
    num_steps: int = 10
    timeout_s: float = 120.0
    save_image_path: Path | None = Path("molmoact2_query_image.jpg")
    save_response_path: Path | None = Path("molmoact2_response.json")


def _as_rgb_uint8_image(value: Any) -> Image.Image:
    array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(f"Expected image array with 3 dimensions, got shape {array.shape}.")

    # Accept either HWC or CHW. LeRobot live cameras normally return HWC.
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 image channels, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _select_image(frame: dict[str, Any], camera_key: str | None) -> tuple[str, Image.Image]:
    image_keys = sorted(key for key in frame if key.startswith("observation.images."))
    if not image_keys:
        raise RuntimeError(f"No observation image keys found. Frame keys: {sorted(frame)}")

    if camera_key is None:
        selected = image_keys[0]
    else:
        selected = camera_key
        if not selected.startswith("observation.images."):
            selected = f"observation.images.{selected}"
        if selected not in frame:
            raise ValueError(
                f"Requested camera key {camera_key!r}, but available image keys are: {image_keys}"
            )

    return selected, _as_rgb_uint8_image(frame[selected])


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach {url}. Check that the SSH tunnel is open and the Brev server is running."
        ) from exc


@parser.wrap()
def query_molmoact2_once(cfg: QueryMolmoAct2OnceConfig) -> None:
    init_logging()

    robot = make_robot_from_config(cfg.robot)

    try:
        print("Connecting robot and camera...")
        robot.connect()

        print("Reading one observation...")
        raw_obs = robot.get_observation()
        features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
        frame = build_dataset_frame(features, raw_obs, prefix=OBS_STR)

        if "observation.state" not in frame:
            raise RuntimeError(f"No observation.state found. Frame keys: {sorted(frame)}")

        state = np.asarray(frame["observation.state"], dtype=np.float32)
        if state.shape != (6,):
            raise RuntimeError(
                f"MolmoAct2-SO100_101 expects 6 state values. Got shape {state.shape}: {state.tolist()}"
            )

        image_key, image = _select_image(frame, cfg.camera_key)
        if cfg.save_image_path is not None:
            cfg.save_image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cfg.save_image_path)

        payload = {
            "task": cfg.task,
            "state": state.tolist(),
            "images": [_image_to_base64_png(image)],
            "num_steps": cfg.num_steps,
        }

        state_names = features["observation.state"]["names"]
        print(f"Task: {cfg.task}")
        print(f"State names: {state_names}")
        print(f"State values: {state.tolist()}")
        print(f"Image key: {image_key}, size: {image.size}")
        if cfg.save_image_path is not None:
            print(f"Saved query image to: {cfg.save_image_path}")

        print(f"Posting observation to: {cfg.server_url}")
        result = _post_json(cfg.server_url, payload, cfg.timeout_s)

        actions = np.asarray(result.get("actions"), dtype=np.float32)
        print(f"Returned action shape: {actions.shape}")
        if actions.size:
            print(f"First action: {actions[0].tolist()}")
            print("Full action chunk:")
            print(json.dumps(actions.tolist(), indent=2))

        if cfg.save_response_path is not None:
            cfg.save_response_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.save_response_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Saved response to: {cfg.save_response_path}")

        print("No action was sent to the robot.")

    finally:
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    try:
        query_molmoact2_once()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
