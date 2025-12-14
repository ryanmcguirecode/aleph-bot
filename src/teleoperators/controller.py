from abc import abstractmethod

from lerobot.teleoperators import Teleoperator


class Controller(Teleoperator):

    @abstractmethod
    def motion_scale(self) -> float:
        """Translation scaling factor for remote -> robot"""
        pass
