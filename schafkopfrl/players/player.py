from abc import ABC, abstractmethod

from rules import Rules


class Player(ABC):

    def __init__(self):
        super().__init__()
        self.rules = Rules()

    @abstractmethod
    def act(self, state):
        pass