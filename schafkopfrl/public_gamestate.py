# contains information about the game that is known by all players
import numpy as np
from schafkopfrl.rules import Rules

class PublicGameState:

    rules = Rules()

    def __init__(self, dealer):
        self.dealer = dealer
        self.first_player = (dealer + 1) % 4

        self.game_stage = self.rules.BIDDING
        self.game_type = [None, None]
        self.game_player = None
        self.trick_number = 0
        self.played_cards = 0

        self.current_player = self.first_player

        # who wants to play what
        self.bidding_round = [None for x in range(4)]

        # who doubled the game (kontra / retour)
        self.contra = [None for x in range(4)]
        self.retour = [None for x in range(4)]

        # cards ordered by players
        self.course_of_game_playerwise = [[[None, None] for x in range(4)] for y in range(8)]

        # cards ordered by the time they were played
        self.course_of_game = [[[None, None] for x in range(4)] for y in range(8)]

        # which player took the trick
        self.trick_owner = [None] * 8

        self.scores = [0, 0, 0, 0]

        #which player is davongelaufen
        self.davongelaufen = None
