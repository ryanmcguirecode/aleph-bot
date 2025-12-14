import os
import shutil
import logging
from this import d
import time
from typing import Any

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.scripts.lerobot_edit_dataset import merge_datasets, delete_episodes

from .experiment import DataCollection
from .keyboard_control import KeyboardControl


log = logging.getLogger("collect")

TEMP_REPO_ID = "dylanmcguir3/temp"


def record_loop(
    robot,
    teleop,
    dataset: LeRobotDataset | None,
    keyboard_control: KeyboardControl,
    fps: int,
    control_time_s: float,
    single_task: str,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
):
    """Custom recording loop with keyboard control."""
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        # Check for early exit
        if keyboard_control.get_and_clear_flag("exit_early"):
            break
        
        # Get robot observation
        obs = robot.get_observation()
        
        # Process observation
        obs_processed = robot_observation_processor(obs)
        
        # Get action from teleop
        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs))
        
        # Process action for robot
        action_values = act_processed_teleop
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        
        # Send action to robot
        robot.send_action(robot_action_to_send)
        
        # Write to dataset if provided (skip during reset)
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        # Maintain fps
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        
        timestamp = time.perf_counter() - start_episode_t


def collect(experiment : DataCollection):
    """Simple, teleop-style main loop for recording episodes."""

    robot = experiment.robot
    teleop = experiment.controller
    repo_id = experiment.repo_id
    fps = experiment.fps
    
    # Dataset features
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    dataset_features = {**obs_features, **action_features}
    
    # # Delete cache directory if it exists
    cache_dir = "/home/dylan/.cache/huggingface/lerobot/"
    if os.path.exists(cache_dir + TEMP_REPO_ID):
        shutil.rmtree(cache_dir + TEMP_REPO_ID)  # Recursively delete the folder
        print(f"Deleted: {cache_dir + TEMP_REPO_ID}")
    else:
        print(f"Path not found: {cache_dir + TEMP_REPO_ID}")
    
    # Check if dataset exists remotely
    api = HfApi()
    exists_remote = False
    exists_remote = api.repo_exists(repo_id, repo_type="dataset")


    if exists_remote:
        dataset = LeRobotDataset(repo_id=repo_id, force_cache_sync=True)

        # Initialize image writer with high-performance settings for recording
        # This is critical: without it, image writing is synchronous and blocks the loop
        dataset.start_image_writer(num_processes=0, num_threads=100)
        # Initialize episode buffer for recording
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=100,
            batch_encoding_size=1,
        )
    
    task = experiment.task_description
    num_episodes = experiment.num_episodes
    target_hz = experiment.target_hz
    episode_time = experiment.episode_time_sec
    reset_time = experiment.reset_time_sec  # Time for reset between episodes

    # Setup keyboard control
    keyboard_control = KeyboardControl()
    keyboard_control.start()

    robot.connect()
    teleop.connect()
    log.info("Recording loop started.")
    log.info("Controls: ESC=exit, LEFT=re-record current episode, RIGHT=save episode, BACKSPACE=skip current episode")

    episode = dataset.num_episodes
    num_episodes = num_episodes + dataset.num_episodes
    
    while episode < num_episodes and not keyboard_control.exit_requested:
        log_say(f"Recording episode {episode}/{num_episodes}")

        try:
            record_loop(
                robot=robot,
                teleop=teleop,
                dataset=dataset,
                keyboard_control=keyboard_control,
                fps=target_hz,
                control_time_s=episode_time,
                single_task=task,
                teleop_action_processor=robot.teleop_action_processor,
                robot_action_processor=robot.robot_action_processor,
                robot_observation_processor=robot.robot_observation_processor,
            )
        except Exception as e:
            log.error(f"Error during episode: {e}")
            continue

        # check for exit early request
        if keyboard_control.get_and_clear_flag("exit_early"):
            log_say("Exiting early...")
            #encode temporary episode videos
            for video_key in dataset.episode_buffer["videos"].keys():
                dataset._encode_temporary_episode_video(video_key, dataset.episode_buffer["episode_index"])
            break

        # Check for re-record request
        if keyboard_control.get_and_clear_flag("rerecord_episode"):
            log_say("Re-recording episode...")
            robot.pause()
            dataset.clear_episode_buffer(delete_images=True)
            print("Clearing episode buffer...")
            robot.resume()
            robot.reset()
            continue

        # Check for save episode request
        if keyboard_control.get_and_clear_flag("save_episode"):
            if not dataset.episode_buffer or dataset.episode_buffer["size"] == 0:
                log_say("No episodes to save")
                robot.pause()
                dataset.clear_episode_buffer(delete_images=True)
                robot.resume()
                continue
            log_say("Saving episode...")
            robot.pause()
            dataset.save_episode()
            print(f"Saved episode {episode}")
            episode += 1
            robot.resume()
            continue
        
        # loop exits due to timeout
        continue

    # Cleanup
    keyboard_control.stop()
    teleop.disconnect()
    robot.disconnect()


    
    dataset.finalize()
    dataset.push_to_hub(repo_id=repo_id, upload_large_folder=True)

    log.info("Shutdown complete.")
