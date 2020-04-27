from abc import ABC, abstractmethod

from schafkopfrl.rules import Rules


class Player(ABC):

    def __init__(self, id):
        self.id = id
        self.cards = []
        self.davongelaufen = False
        self.rules = Rules()
        super().__init__()

    def take_cards(self, cards):
        self.cards = cards
        self.davongelaufen = False

    @abstractmethod
    def call_game_type(self, game_state):
        pass

    @abstractmethod
    def contra_retour(self, game_state):
        pass

    @abstractmethod
    def select_card(self, game_state):
        pass

    @abstractmethod
    def retrieve_reward(self, reward, game_state):
        pass