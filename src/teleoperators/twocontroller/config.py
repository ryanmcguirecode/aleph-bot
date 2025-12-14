from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators import Teleoperator


@dataclass
class TwoControllerConfig(TeleoperatorConfig):
    id = "0"
    left: Teleoperator
    right: Teleoperator
    left_config: TeleoperatorConfig
    right_config: TeleoperatorConfig
