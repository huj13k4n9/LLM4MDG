from abc import ABC, abstractmethod
from typing import Any


class Action(ABC):
    """
    A base class for handling different actions.
    """

    @abstractmethod
    def run(self) -> Any:
        pass
