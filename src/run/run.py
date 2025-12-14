import argparse
import logging
from enum import Enum

from cameraSpec import CAMERAS_DICT
from robots.twoarm.config import TwoArmConfig
from robots.twoarm.twoarm import TwoArm
from run.collect import collect
from teleoperators.twocontroller.config import TwoControllerConfig
from teleoperators.twocontroller.twocontroller import TwoController

from .experiment import DataCollection, ExperimentFactory, Teleop, Experiment

from robots.xarm.robot import XArmRobot
from robots.xarm.lite6.config import Lite6Config
from robots.xarm.xarm7.config import XArm7Config
from .teleop import teleop

from teleoperators.omni.omni import OmniTeleoperator
from teleoperators.omni.config import OmniConfig

TARGET_HZ = 100

experiments: dict[str, ExperimentFactory] = {
    "COLLECT": lambda: DataCollection(
        name="COLLECT",
        robot=XArmRobot(
             Lite6Config(start_pos=(5.5, 29, 24, 7.5, -70.4, -3.1), cameras=CAMERAS_DICT, dt=1.0 / TARGET_HZ)
        ),
        controller=OmniTeleoperator(OmniConfig(translation_scales=(0.1, 0.4, 1.0), scale_rotation=1.0, device_name="USB", cutoff_frequency_hz=7, sampling_rate_hz=TARGET_HZ)),
        target_hz=TARGET_HZ,
        fps=TARGET_HZ,
        num_episodes=25,
        episode_time_sec=600,
        reset_time_sec=0.5,
        task_description="Insert tool into vessel",
        dataset_name="xarm7-collect-encoded",
        repo_id="dylanmcguir3/xarm7-collect",
    ),
    "XARM7": lambda: Teleop(
        name="XARM7",
        robot=XArmRobot(
            XArm7Config(start_pos=(1.7, -8.7, 17, 26, 103, 37, 27), cameras={}, dt=1.0 / TARGET_HZ)
        ),
        controller=OmniTeleoperator(
            OmniConfig(
                translation_scales=(0.1, 0.4, 1.0),
                scale_rotation=1.0,
                device_name="USB",
                cutoff_frequency_hz=12,
                sampling_rate_hz=TARGET_HZ
            )
        ),
        target_hz=TARGET_HZ,
    ),
    "LITE6": lambda: Teleop(
        name="LITE6",
        robot=XArmRobot(
            Lite6Config(start_pos=(5.5, 29, 24, 7.5, -70.4, -3.1), cameras=CAMERAS_DICT, dt=1.0 / TARGET_HZ)
        ),
        controller=OmniTeleoperator(OmniConfig(translation_scales=(0.1, 0.4, 1.0), scale_rotation=1.0, device_name="LAN", cutoff_frequency_hz=5, sampling_rate_hz=TARGET_HZ)),
        target_hz=TARGET_HZ,
    ),
    "LITE6-USB": lambda: Teleop(
        name="LITE6",
        robot=XArmRobot(
            Lite6Config(start_pos=(5.5, 29, 24, 7.5, -70.4, -3.1), cameras=CAMERAS_DICT, dt=1.0 / TARGET_HZ)
        ),
        controller=OmniTeleoperator(OmniConfig(translation_scales=(0.1, 0.4, 1.0), scale_rotation=1.0, device_name="USB", cutoff_frequency_hz=5, sampling_rate_hz=TARGET_HZ)),
        target_hz=TARGET_HZ,
    ),
    "2ROBOTS": lambda: Teleop(
        name="2ROBOTS",
        robot=TwoArm(
            TwoArmConfig(
                left=XArmRobot,
                right=XArmRobot,
                left_config=Lite6Config(
                    start_pos=(-59, 29, 56, 11.7, -24.2, -11), cameras={}, dt=1.0 / TARGET_HZ
                ),
                right_config=XArm7Config(
                    start_pos=(1.7, -8.7, 17, 26, 103, 37, 27), cameras={}, dt=1.0 / TARGET_HZ
                ),
                cameras={},
            )
        ),
        controller=TwoController(
            TwoControllerConfig(
                left=OmniTeleoperator,
                right=OmniTeleoperator,
                left_config=OmniConfig(
                    translation_scales=(0.1, 0.4, 1.0),
                    scale_rotation=1.0,
                    device_name="LAN",
                    id="0",
                    buttons=("GRIPPER", "SCALE"),
                    cutoff_frequency_hz=20,
                    sampling_rate_hz=TARGET_HZ
                ),
                right_config=OmniConfig(
                    translation_scales=(0.1, 0.4, 1.0),
                    scale_rotation=1.0,
                    device_name="USB",
                    id="1",
                    buttons=("GRIPPER", "KILL"),
                    cutoff_frequency_hz=20,
                    sampling_rate_hz=TARGET_HZ
                ),
            )
        ),
        target_hz=TARGET_HZ,
    ),
}


class Mode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TELEOP = "teleop"
    COLLECT = "collect"

    def __str__(self):
        return self.value


def run():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run Aleph experiment")
    parser.add_argument(
        "--mode", choices=list(Mode), default=Mode.TELEOP, type=Mode, help="Run mode"
    )
    parser.add_argument(
        "--experiment", type=str, default="TELEOP", help="Experiment name"
    )

    args = parser.parse_args()

    factory: ExperimentFactory | None = experiments.get(args.experiment.upper())
    if factory is None:
        raise ValueError(
            f"Experiment {args.experiment} not found. Available: {list(experiments)}"
        )

    experiment: Experiment = factory()
    if experiment is None:
        raise ValueError(
            f"Experiment {args.experiment} not found. Available: {[e.name for e in experiments.values()]}"
        )

    logging.info(f"Running {experiment.name} with mode={args.mode}")

    if args.mode == Mode.TELEOP:
        if not isinstance(experiment, Teleop):
            raise TypeError(
                f"teleop() only accepts Teleop or subclasses, got {type(experiment).__name__}"
            )
        teleop(experiment)
    elif args.mode == Mode.TRAIN:
        raise NotImplementedError("Train not implemented")
    elif args.mode == Mode.INFERENCE:
        raise NotImplementedError("Inference not implemented")
    elif args.mode == Mode.COLLECT:
        if not isinstance(experiment, DataCollection):
            raise TypeError(
                f"collect() only accepts DataCollection or subclasses, got {type(experiment).__name__}"
            )
        collect(experiment)
    else:
        raise NotImplementedError(args.mode + " not implemented")


if __name__ == "__main__":
    run()
