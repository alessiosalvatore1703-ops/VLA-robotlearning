#!/usr/bin/env python3
"""LeRobot recording with an automatic start pose before every episode.

This is a thin variant of `lerobot-record`. It keeps the same LeRobot dataset
format and command-line robot/dataset/teleop arguments, but adds:

    --start_pose_path=start_pose.json
    --start_pose_move_time_s=2.0
    --start_pose_hold_time_s=1.0

The robot is moved to the pose before each recorded episode. The move itself is
not saved in the dataset.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import (
    KeyboardTeleop,
    LeRobotDataset,
    RecordConfig,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    Teleoperator,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    build_dataset_frame,
    combine_feature_dicts,
    create_initial_features,
    init_keyboard_listener,
    init_logging,
    init_rerun,
    is_headless,
    log_rerun_data,
    log_say,
    make_default_processors,
    make_robot_from_config,
    make_teleoperator_from_config,
    record_loop,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class RecordWithStartPoseConfig(RecordConfig):
    # JSON produced by collection/capture_start_pose.py.
    start_pose_path: Path | None = None
    # Time used to interpolate from the current robot pose to the start pose.
    start_pose_move_time_s: float = 2.0
    # Time to hold the start pose before recording begins.
    start_pose_hold_time_s: float = 1.0
    # If true, stop if the pose JSON does not match the robot motor keys.
    strict_start_pose_keys: bool = True
    # Move the leader/teleop arm to the same start pose before each episode.
    move_teleop_to_start_pose: bool = True
    # Release leader torque after moving it so it can be used for teleoperation.
    release_teleop_after_start_pose: bool = True


def _load_start_pose(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    pose: dict[str, float] = {}
    for key, value in raw.items():
        pose_key = key if key.endswith(".pos") else f"{key}.pos"
        pose[pose_key] = float(value)
    return pose


def _current_motor_pose(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {key: float(value) for key, value in obs.items() if key.endswith(".pos")}


def _validate_pose_keys(
    *,
    label: str,
    current: dict[str, float],
    start_pose: dict[str, float],
    strict_keys: bool,
) -> dict[str, float]:
    if not current:
        raise RuntimeError(f"{label} did not contain any '*.pos' motor keys.")

    missing = sorted(set(start_pose) - set(current))
    extra = sorted(set(current) - set(start_pose))
    if missing:
        raise ValueError(f"Start pose contains keys not present on {label}: {missing}")
    if strict_keys and extra:
        raise ValueError(
            f"Start pose is missing {label} motor keys: "
            f"{extra}. Capture a new pose or set --strict_start_pose_keys=false."
        )

    target = {key: start_pose[key] for key in current if key in start_pose}
    if not target:
        raise RuntimeError(f"No overlapping motor keys between {label} and start pose.")
    return target


def move_to_start_pose(
    robot,
    start_pose: dict[str, float] | None,
    *,
    move_time_s: float,
    hold_time_s: float,
    fps: int,
    strict_keys: bool,
) -> None:
    """Move the robot to the configured pose without recording dataset frames."""
    if start_pose is None:
        return

    current = _current_motor_pose(robot)
    target = _validate_pose_keys(
        label="robot",
        current=current,
        start_pose=start_pose,
        strict_keys=strict_keys,
    )

    steps = max(1, int(move_time_s * fps))
    interval_s = 1.0 / fps

    logging.info("Moving robot to start pose over %.2fs", move_time_s)
    for step in range(1, steps + 1):
        alpha = step / steps
        action = {
            key: current[key] + (target[key] - current[key]) * alpha
            for key in target
        }
        robot.send_action(action)
        precise_sleep(interval_s)

    robot.send_action(target)
    if hold_time_s > 0:
        time.sleep(hold_time_s)


def move_teleop_to_start_pose(
    teleop,
    start_pose: dict[str, float] | None,
    *,
    move_time_s: float,
    hold_time_s: float,
    fps: int,
    strict_keys: bool,
    release_after_move: bool,
) -> None:
    """Move a leader arm to the configured pose, then optionally release torque."""
    if teleop is None or start_pose is None:
        return
    if not hasattr(teleop, "get_action") or not hasattr(teleop, "send_feedback"):
        logging.warning("Teleoperator does not support pose feedback; not moving it to start pose.")
        return
    if not hasattr(teleop, "enable_torque") or not hasattr(teleop, "disable_torque"):
        logging.warning("Teleoperator does not expose torque control; not moving it to start pose.")
        return

    current = {key: float(value) for key, value in teleop.get_action().items() if key.endswith(".pos")}
    target = _validate_pose_keys(
        label="teleop",
        current=current,
        start_pose=start_pose,
        strict_keys=strict_keys,
    )

    steps = max(1, int(move_time_s * fps))
    interval_s = 1.0 / fps

    logging.info("Moving teleop to start pose over %.2fs", move_time_s)
    teleop.enable_torque()
    try:
        for step in range(1, steps + 1):
            alpha = step / steps
            feedback = {
                key: current[key] + (target[key] - current[key]) * alpha
                for key in target
            }
            teleop.send_feedback(feedback)
            precise_sleep(interval_s)

        teleop.send_feedback(target)
        if hold_time_s > 0:
            time.sleep(hold_time_s)
    finally:
        if release_after_move:
            teleop.disable_torque()


@parser.wrap()
def record_with_start_pose(
    cfg: RecordWithStartPoseConfig,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation] | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    start_pose = _load_start_pose(cfg.start_pose_path)
    if start_pose is None:
        logging.warning("No --start_pose_path provided; behavior is identical to lerobot-record.")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "Use a non-eval dataset name for recording."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. Consider enabling "
                "--dataset.streaming_encoding=true --dataset.encoder_threads=2 for faster episode saving."
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say("Moving to start pose", cfg.play_sounds)
                move_to_start_pose(
                    robot,
                    start_pose,
                    move_time_s=cfg.start_pose_move_time_s,
                    hold_time_s=0.0,
                    fps=cfg.dataset.fps,
                    strict_keys=cfg.strict_start_pose_keys,
                )
                if cfg.move_teleop_to_start_pose:
                    move_teleop_to_start_pose(
                        teleop,
                        start_pose,
                        move_time_s=cfg.start_pose_move_time_s,
                        hold_time_s=cfg.start_pose_hold_time_s,
                        fps=cfg.dataset.fps,
                        strict_keys=cfg.strict_start_pose_keys,
                        release_after_move=cfg.release_teleop_after_start_pose,
                    )
                elif cfg.start_pose_hold_time_s > 0:
                    time.sleep(cfg.start_pose_hold_time_s)

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved; skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main() -> None:
    register_third_party_plugins()
    record_with_start_pose()


if __name__ == "__main__":
    main()
