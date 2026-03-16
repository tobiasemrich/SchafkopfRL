# contains information about the game that is known by all players
from typing import Optional

from .rules import Rules

Card = tuple[Optional[int], Optional[int]]
GameType = tuple[Optional[int], Optional[int]]

_NONE_CARD: Card = Rules.NONE_CARD

class PublicGameState:
    """Publicly visible game state shared among all players.

    Tracks all information that is known to every player: game stage,
    bidding results, played cards, trick ownership, scores, and
    Contra/Retour declarations. Uses ``__slots__`` for memory efficiency.

    Parameters
    ----------
    dealer : int
        Index (0–3) of the dealer for this round.
    """
    __slots__ = ('dealer', 'first_player', 'game_stage', 'game_type', 'game_player',
                 'trick_number', 'played_cards', 'current_player', 'bidding_round',
                 'contra', 'retour', 'course_of_game_playerwise', 'course_of_game',
                 'trick_owner', 'scores', 'davongelaufen', 'action_probabilities')

    def __init__(self, dealer: int) -> None:
        self.dealer: int = dealer
        self.first_player: int = (dealer + 1) % 4

        self.game_stage: int = Rules.BIDDING
        self.game_type: GameType = (None, None)
        self.game_player: Optional[int] = None
        self.trick_number: int = 0
        self.played_cards: int = 0

        self.current_player: int = self.first_player

        # who wants to play what
        self.bidding_round: list[Optional[GameType]] = [None, None, None, None]

        # who doubled the game (kontra / retour)
        self.contra: list[Optional[bool]] = [None, None, None, None]
        self.retour: list[Optional[bool]] = [None, None, None, None]

        # cards ordered by players
        self.course_of_game_playerwise: list[list[Card]] = [[_NONE_CARD, _NONE_CARD, _NONE_CARD, _NONE_CARD] for _ in range(8)]

        # cards ordered by the time they were played
        self.course_of_game: list[list[Card]] = [[_NONE_CARD, _NONE_CARD, _NONE_CARD, _NONE_CARD] for _ in range(8)]

        # which player took the trick
        self.trick_owner: list[Optional[int]] = [None] * 8

        self.scores: list[int] = [0, 0, 0, 0]

        #which player is davongelaufen
        self.davongelaufen: Optional[int] = None

        # for debugging purposes remember probs for picking an action
        self.action_probabilities: list[list[Optional[float]]] = [[None, None, None, None] for _ in range(11)]

    def __deepcopy__(self, memo: dict) -> "PublicGameState":
        """Create a deep copy of this game state.

        Parameters
        ----------
        memo : dict
            Memoization dict used by ``copy.deepcopy``.

        Returns
        -------
        PublicGameState
            Independent copy of the current state.
        """
        cls = self.__class__
        result: "PublicGameState" = cls.__new__(cls)

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