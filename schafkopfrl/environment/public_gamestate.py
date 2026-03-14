# contains information about the game that is known by all players
from .rules import Rules

_NONE_CARD = Rules.NONE_CARD

class PublicGameState:
    __slots__ = ('dealer', 'first_player', 'game_stage', 'game_type', 'game_player',
                 'trick_number', 'played_cards', 'current_player', 'bidding_round',
                 'contra', 'retour', 'course_of_game_playerwise', 'course_of_game',
                 'trick_owner', 'scores', 'davongelaufen', 'action_probabilities')

    def __init__(self, dealer):
        self.dealer = dealer
        self.first_player = (dealer + 1) % 4

        self.game_stage = Rules.BIDDING
        self.game_type = (None, None)
        self.game_player = None
        self.trick_number = 0
        self.played_cards = 0

        self.current_player = self.first_player

        # who wants to play what
        self.bidding_round = [None, None, None, None]

        # who doubled the game (kontra / retour)
        self.contra = [None, None, None, None]
        self.retour = [None, None, None, None]

        # cards ordered by players
        self.course_of_game_playerwise = [[_NONE_CARD, _NONE_CARD, _NONE_CARD, _NONE_CARD] for _ in range(8)]

        # cards ordered by the time they were played
        self.course_of_game = [[_NONE_CARD, _NONE_CARD, _NONE_CARD, _NONE_CARD] for _ in range(8)]

        # which player took the trick
        self.trick_owner = [None] * 8

        self.scores = [0, 0, 0, 0]

        #which player is davongelaufen
        self.davongelaufen = None

        # for debugging purposes remember probs for picking an action
        self.action_probabilities = [[None, None, None, None] for _ in range(11)]

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
        result.bidding_round = self.bidding_round[:]
        result.contra = self.contra[:]
        result.retour = self.retour[:]
        result.course_of_game_playerwise = [self.course_of_game_playerwise[y][:] for y in range(8)]
        result.course_of_game = [self.course_of_game[y][:] for y in range(8)]
        result.trick_owner = self.trick_owner[:]
        result.scores = self.scores[:]
        result.davongelaufen = self.davongelaufen
        result.action_probabilities = self.action_probabilities

        return result