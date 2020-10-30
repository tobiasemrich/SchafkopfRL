# contains information about the game that is known by all players
from copy import copy

from rules import Rules

class PublicGameState:


    def __init__(self, dealer):
        self.dealer = dealer
        self.first_player = (dealer + 1) % 4

        self.game_stage = Rules.BIDDING
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

        # for debugging purposes remember probs for picking an action
        self.action_probabilities = [[None for x in range(4)] for y in range(11)]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        result.dealer = self.dealer
        result.first_player = self.first_player
        result.game_stage = self.game_stage
        result.game_type = self.game_type
        result.game_player = self.game_player
        result.trick_number = self.trick_number
        result.played_cards = self.played_cards
        result.current_player = self.current_player
        result.bidding_round = copy(self.bidding_round)
        result.contra = copy(self.contra)
        result.retour = copy(self.retour)
        result.course_of_game_playerwise = [copy(self.course_of_game_playerwise[y]) for y in range(8)]
        result.course_of_game = [copy(self.course_of_game[y]) for y in range(8)]
        result.trick_owner = copy(self.trick_owner)
        result.scores = copy(self.scores)
        result.davongelaufen = self.davongelaufen
        result.action_probabilities = self.action_probabilities

        return result