from dataclasses import dataclass
from typing import Callable

from lerobot.robots import Robot

from teleoperators.controller import Controller


@dataclass
class Experiment:
    name: str
    robot: Robot
    description: str = "Generic Experiment"


ExperimentFactory = Callable[[], Experiment]


@dataclass(kw_only=True)
class Teleop(Experiment):
    controller: "Controller"
    target_hz: float = 250.0

    def __post_init__(self):
        self.description = "Teleop"


@dataclass(kw_only=True)
class DataCollection(Experiment):
    controller: "Controller"
    num_episodes: int = 25
    target_hz: float = 250.0
    fps: int = 30
    episode_time_sec: float = 60
    task_description: str = "Insert tool into vessel"
    dataset_name: str = "insert-tool"
    repo_id: str = f"dylanmcguir3/{dataset_name}"
    reset_time_sec: float = 2.0

    def __post_init__(self):
        self.description = "Data Collection"
