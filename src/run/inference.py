import os
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from typing_extensions import Final

from meca.robot import MecaRobot
from meca.config import MecaConfig

# -------------------------
# User settings
# -------------------------
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 100
TASK_DESCRIPTION = "Microsurgery teleop task"

HF_USER = "dylanmcguir3"
MODEL_ID = "insert-tool-128"  # subfolder for this specific task
MODEL_TYPE = "diffusion"
CHECKPOINT = "100000"

REST_JOINTS: Final[tuple[float, float, float, float, float, float]] = (
    0,
    60,
    20,
    0,
    -60.0,
    0,
)

robot_config = MecaConfig(
    ip="192.168.0.100", id="meca_eval_robot", start_pos=REST_JOINTS
)
robot = MecaRobot(robot_config)

# -------------------------
# Load trained policy
# -------------------------
if MODEL_TYPE == "diffusion":
    policy = DiffusionPolicy.from_pretrained(
        Path(
            f"/home/dylan/surgery-automation/outputs/train/{MODEL_ID}/checkpoints/{CHECKPOINT}/pretrained_model"
        )
    )
elif MODEL_TYPE == "act":
    policy = ACTPolicy.from_pretrained(
        Path(
            f"/home/dylan/surgery-automation/outputs/train/{MODEL_ID}/checkpoints/{CHECKPOINT}/pretrained_model"
        )
    )

# -------------------------
# Dataset features
# -------------------------
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# delete files from .cache/huggingface/datasets/dylanmcguir3-meca-needle-pick-diffusion-v2-eval
# Path to your Hugging Face dataset cache
cache_dir = f"/home/dylan/.cache/huggingface/lerobot/{HF_USER}/{MODEL_ID}-eval"

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)  # Recursively delete the folder
    print(f"Deleted: {cache_dir}")
else:
    print(f"Path not found: {cache_dir}")

dataset = LeRobotDataset.create(
    repo_id=cache_dir,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# -------------------------
# Keyboard + visualization
# -------------------------
_, events = init_keyboard_listener()
init_rerun(session_name="policy_eval")

# -------------------------
# Connect robot
# -------------------------
robot.connect()

# Pre/post processors for policy I/O
preprocessor, postprocessor = make_pre_post_processors(
    policy,
    pretrained_path=f"/home/dylan/surgery-automation/outputs/train/{MODEL_ID}/checkpoints/{CHECKPOINT}/pretrained_model",
    dataset_stats=dataset.meta.stats,
)

# -------------------------
# Evaluation loop
# -------------------------
for episode_idx in range(NUM_EPISODES):
    log_say(
        f"Running inference, recording eval episode {episode_idx + 1}/{NUM_EPISODES}"
    )

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=robot.teleop_action_processor,
        robot_action_processor=robot.robot_action_processor,
        robot_observation_processor=robot.robot_observation_processor,
    )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        robot.reset_position()
        continue

    dataset.save_episode()

# -------------------------
# Cleanup
# -------------------------
robot.disconnect()
dataset.push_to_hub()
