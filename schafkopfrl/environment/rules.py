from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .public_gamestate import PublicGameState

Card = tuple[Optional[int], Optional[int]]
GameType = tuple[Optional[int], Optional[int]]

class Rules:
    """Rules engine for the Bavarian card game Schafkopf.

    Contains all game rules including trump ordering, allowed actions
    (bidding, contra/retour, card play), trick evaluation, and scoring.
    Used by environments and policies to validate and evaluate game actions.
    """
    NONE_CARD: Card = (None, None)

    # phases of the game
    BIDDING: int = 1
    CONTRA: int = 2
    RETOUR: int = 3
    TRICK: int = 4

    #only for efficiency — tuples for O(1) frozenset membership
    SAUSPIEL_TRUMPS = ((1, 0), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4),
              (3, 4))
    SCHELLENSOLO_TRUMPS = ((0, 0), (0, 1), (0, 2), (0, 5), (0, 6), (0, 7), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4),
              (3, 4))
    HERZSOLO_TRUMPS = ((1, 0), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4))
    GRASSOLOTRUMPS = ((2, 0), (2, 1), (2, 2), (2, 5), (2, 6), (2, 7), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4))
    EICHELSOLO_TRUMPS = ((3, 0), (3, 1), (3, 2), (3, 5), (3, 6), (3, 7), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4))
    WENZ_TRUMPS = ((0, 3), (1, 3), (2, 3), (3, 3))

    # frozensets for O(1) membership testing
    SAUSPIEL_TRUMPS_SET = frozenset(SAUSPIEL_TRUMPS)
    SCHELLENSOLO_TRUMPS_SET = frozenset(SCHELLENSOLO_TRUMPS)
    HERZSOLO_TRUMPS_SET = frozenset(HERZSOLO_TRUMPS)
    GRASSOLOTRUMPS_SET = frozenset(GRASSOLOTRUMPS)
    EICHELSOLO_TRUMPS_SET = frozenset(EICHELSOLO_TRUMPS)
    WENZ_TRUMPS_SET = frozenset(WENZ_TRUMPS)

    def __init__(self) -> None:
        self.card_number: list[str] = ['siebener',
                            'achter',
                            'neuner',
                            'unter',
                            'ober',
                            'koenig',
                            'zehner',
                            'sau']

        self.card_color: list[str] = ['schellen', 'herz', 'gras', 'eichel']

        self.card_scores: list[int] = [0, 0, 0, 2, 3, 4, 10, 11]

        ############## schelle # herz # gras # eichel #
        self.cards: list[tuple[int, int]] = [(0, 0), (1, 0), (2, 0), (3, 0),  # siebener
                      (0, 1), (1, 1), (2, 1), (3, 1),  # achter
                      (0, 2), (1, 2), (2, 2), (3, 2),  # neuner
                      (0, 3), (1, 3), (2, 3), (3, 3),  # unter
                      (0, 4), (1, 4), (2, 4), (3, 4),  # ober
                      (0, 5), (1, 5), (2, 5), (3, 5),  # koenig
                      (0, 6), (1, 6), (2, 6), (3, 6),  # zehner
                      (0, 7), (1, 7), (2, 7), (3, 7)]  # sau

        self.game_names: list[str] = ['sauspiel', 'wenz', 'solo']

        ############# schelle # herz # gras # eichel #
        self.games: list[GameType] = [(None, None),  # no game
                      (0, 0), (2, 0), (3, 0),  # sauspiel
                      (None, 1),  # wenz
                      (0, 2), (1, 2), (2, 2), (3, 2)]  # solo

        self.reward_basic: list[int] = [0, 20, 50, 50]  # no game, sauspiel, solo, wenz
        self.reward_schneider: list[int] = [0, 10, 20]  # normal, schneider, schneider schwarz
        self.winning_thresholds: list[int] = [0, 30, 60, 90, 119]

        self.reward_laufende: int = 10
        self.min_laufende: list[int] = [3, 2, 3]  # sauspiel, wenz, solo

    def higher_card(self, game_type: GameType, card1: Card, card2: Card) -> bool:
        """Check whether card2 beats card1 in the same trick.

        Assumes card1 was played before card2 in the same trick.

        Parameters
        ----------
        game_type : GameType
            The current game type being played.
        card1 : Card
            The first played card.
        card2 : Card
            The second played card.

        Returns
        -------
        bool
            True if card2 is higher than card1, False otherwise.
        """
        trumps: tuple[Card, ...] = self.get_sorted_trumps(game_type)
        c1_trump: bool = card1 in trumps
        c2_trump: bool = card2 in trumps
        if not c1_trump:
            if not c2_trump:
                if card2[0] != card1[0] or card2[1] < card1[1]:  # not lead color or smaller value
                    return False
                else:
                    return True
            else:
                return True
        else:
            if not c2_trump:
                return False
            else:  # both cards are trumps
                if trumps.index(card1) < trumps.index(card2):
                    return True
                else:
                    return False

    def get_trump_set(self, game_type: GameType) -> frozenset[Card]:
        """Return the frozenset of trump cards for the given game type.

        Parameters
        ----------
        game_type : GameType
            The current game type.

        Returns
        -------
        frozenset[Card]
            Set of trump cards for O(1) membership testing.
        """
        if game_type[1] == 0:  # Sauspiel
            return self.SAUSPIEL_TRUMPS_SET
        elif game_type[1] == 2:  # Solo
            if game_type[0] == 0:
                return self.SCHELLENSOLO_TRUMPS_SET
            elif game_type[0] == 1:
                return self.HERZSOLO_TRUMPS_SET
            elif game_type[0] == 2:
                return self.GRASSOLOTRUMPS_SET
            elif game_type[0] == 3:
                return self.EICHELSOLO_TRUMPS_SET
        else:  # wenz
            return self.WENZ_TRUMPS_SET

    def get_sorted_trumps(self, game_type: GameType) -> tuple[Card, ...]:
        """Return trump cards sorted in ascending order of strength.

        Parameters
        ----------
        game_type : GameType
            The current game type.

        Returns
        -------
        tuple[Card, ...]
            Trump cards ordered from weakest to strongest.
        """
        if game_type[1] == 0:  # Sauspiel
            #trump_colors = [1]  # Herz
            #trump_numbers = [3, 4]  # Unter, Ober
            return self.SAUSPIEL_TRUMPS

        elif game_type[1] == 2:  # Solo
            #trump_colors = [game_type[0]]
            #trump_numbers = [3, 4]
            if game_type[0] == 0:
                return self.SCHELLENSOLO_TRUMPS
            elif game_type[0] == 1:
                return self.HERZSOLO_TRUMPS
            elif game_type[0] == 2:
                return self.GRASSOLOTRUMPS
            elif game_type[0] == 3:
                return self.EICHELSOLO_TRUMPS
        else:  # wenz
            #trump_colors = []
            #trump_numbers = [3]
            return self.WENZ_TRUMPS

        #trumps_color = [[color, number] for color, number in self.cards if color in trump_colors and number not in trump_numbers]
        #trumps_number = [[color, number] for color, number in self.cards if number in trump_numbers]

        #return trumps_color + trumps_number


    def allowed_actions(self, game_state: "PublicGameState", player_cards: list[Card]) -> list:
        """Return the list of allowed actions for the current player.

        Dispatches to the appropriate method based on the current game stage.

        Parameters
        ----------
        game_state : PublicGameState
            The current public game state.
        player_cards : list[Card]
            The current player's hand.

        Returns
        -------
        list
            Allowed actions (games, booleans, or cards depending on stage).
        """
        if game_state.game_stage == Rules.BIDDING:
            return self.allowed_games(player_cards)
        elif game_state.game_stage == Rules.CONTRA or game_state.game_stage == Rules.RETOUR:
            return self.allowed_contra_retour(game_state, player_cards)
        else:
            return self.allowed_cards(game_state, player_cards)


    def allowed_games(self, player_cards: list[Card]) -> list[GameType]:
        """Return the list of games a player is allowed to bid.

        All games are allowed except Sauspiel with a color the player
        does not hold or already holds the ace of.

        Parameters
        ----------
        player_cards : list[Card]
            The player's hand.

        Returns
        -------
        list[GameType]
            Allowed game types the player may bid.
        """
        player_cards_set: set[Card] = set(player_cards)
        playable_colors: set[int] = {color for color, number in player_cards if
                           number != 3 and  # unter
                           number != 4 and  # ober
                           color != 1 and  # herz
                           (color, 7) not in player_cards_set}  # not the ace
        return [g for g in self.games if g[1] != 0 or g[0] in playable_colors]

    def allowed_cards(self, game_state: "PublicGameState", player_cards: list[Card]) -> list[Card]:
        """Return the cards a player is allowed to play in the current trick.

        Takes into account the player's hand, position in the trick,
        the first card played, the game type, and Davonlaufen status.

        Parameters
        ----------
        game_state : PublicGameState
            The current public game state.
        player_cards : list[Card]
            The current player's hand.

        Returns
        -------
        list[Card]
            Cards the player is allowed to play.
        """
        allowed_cards: list[Card] = []

        trump_set: frozenset[Card] = self.get_trump_set(game_state.game_type)
        rufsau: Card = (game_state.game_type[0], 7)  # might be invalid if a solo is played

        first_player_of_trick = game_state.first_player if game_state.trick_number == 0 else game_state.trick_owner[
            game_state.trick_number - 1]
        if game_state.current_player == first_player_of_trick:  # first player in this trick
            allowed_cards = player_cards.copy()
            # exception is the Rufsau color

            if game_state.game_type[1] == 0 and rufsau in player_cards and not game_state.current_player == game_state.davongelaufen:
                ruf_sau_color_cards = [card for card in player_cards if
                                       (card[0] == game_state.game_type[0] and card not in trump_set and card != rufsau)]
                if len(ruf_sau_color_cards) < 3:
                    for c in ruf_sau_color_cards:
                        allowed_cards.remove(c)
        else:
            first_card = game_state.course_of_game_playerwise[game_state.trick_number][first_player_of_trick]
            if first_card in trump_set:
                player_trumps = [card for card in player_cards if card in trump_set]
                if len(player_trumps) > 0:
                    allowed_cards = player_trumps
                else:
                    allowed_cards = player_cards.copy()
            else:  # color of first card not trump
                if game_state.game_type[1] == 0 and game_state.game_type[0] == first_card[
                    0] and rufsau in player_cards and not game_state.current_player == game_state.davongelaufen:
                    # if the player has the Suchsau and the color is played and he has not davongelaufen then he has to play the ace
                    allowed_cards = [rufsau]
                else:
                    player_first_color_cards = [card for card in player_cards if
                                                card[0] == first_card[0] and card not in trump_set]
                    if len(player_first_color_cards) > 0:
                        allowed_cards = player_first_color_cards
                    else:
                        allowed_cards = player_cards.copy()
            # remove rufsau if not gesucht and not davongelaufen and not last trick
            if game_state.game_type[1] == 0 and rufsau in allowed_cards and not (first_card[0] == rufsau[0] or game_state.current_player == game_state.davongelaufen or game_state.trick_number == 7):
                allowed_cards.remove(rufsau)

        return allowed_cards

    def allowed_contra_retour(self, game_state: "PublicGameState", player_cards: list[Card]) -> list[bool]:
        """Return whether the player may double (contra/retour) the game.

        Parameters
        ----------
        game_state : PublicGameState
            The current public game state.
        player_cards : list[Card]
            The current player's hand.

        Returns
        -------
        list[bool]
            ``[False, True]`` if doubling is allowed, ``[False]`` otherwise.
        """
        allowed: list[bool] = [False]

        if not any(game_state.contra) and game_state.game_stage == Rules.CONTRA:  # contra check
            allowed.append(True)
            # not allowed if you are the player or the team mate of the player
            if game_state.game_player == game_state.current_player or (
                    game_state.game_type[1] == 0 and ((game_state.game_type[0], 7) in player_cards)):
                allowed = [False]
        elif any(game_state.contra) and not any(game_state.retour) and game_state.game_stage == Rules.RETOUR:  # retour check
            allowed = [False]
            # allowed if you are the player or the team mate of the player
            if game_state.game_player == game_state.current_player or (
                    game_state.game_type[1] == 0 and (game_state.game_type[0], 7) in player_cards):
                allowed.append(True)

        return allowed

    def highest_game(self, bidding_round: list[Optional[GameType]], first_player: int) -> tuple[Optional[int], GameType]:
        """Determine the winning bid from the bidding round.

        Parameters
        ----------
        bidding_round : list[Optional[GameType]]
            Each player's bid (or ``(None, None)`` for pass).
        first_player : int
            Index of the player who bids first.

        Returns
        -------
        tuple[Optional[int], GameType]
            ``(game_player (index), game_type)`` of the highest bid.
        """
        current_highest_game: GameType = (None, None)
        game_player: Optional[int] = None
        for p in range(4):
            player_id = (first_player + p) % 4
            game_type = bidding_round[player_id]
            if current_highest_game[1] == None or (not game_type[1] == None and game_type[1] > current_highest_game[1]):
                current_highest_game = game_type
                game_player = player_id
        return (game_player, current_highest_game)

    def trick_owner(self, trick: list[Card], first_player: int, game_type: GameType) -> int:
        """Return the player who won the trick.

        Parameters
        ----------
        trick : list[Card]
            Cards played in this trick, indexed by player id.
        first_player : int
            Index of the player who led the trick.
        game_type : GameType
            The current game type.

        Returns
        -------
        int
            Player index of the trick winner.
        """
        highest_card_index: int = first_player
        for i in range(1, 4):
            player_id = (first_player + i) % 4
            if self.higher_card(game_type, trick[highest_card_index], trick[player_id]):
                highest_card_index = player_id
        return highest_card_index

    def count_points(self, trick_cards: list[Card]) -> int:
        """Return the total point value of the cards in a trick.

        Parameters
        ----------
        trick_cards : list[Card]
            The four cards played in the trick.

        Returns
        -------
        int
            Sum of card point values.
        """
        return sum([self.card_scores[number] for color, number in trick_cards])