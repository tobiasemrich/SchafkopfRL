from typing import Any

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.torch_utils import FLOAT_MIN

from schafkopfrl.environment.utils import *
from schafkopfrl.environment.rules import Rules, Card
from schafkopfrl.environment.schafkopf_env import SchafkopfEnv
from schafkopfrl.environment.public_gamestate import PublicGameState
import torch.nn as nn
import torch
import random

from schafkopfrl.policy.mcts.mct import MonteCarloTree

_NONE_CARD: Card = Rules.NONE_CARD

class PIMCModule(RLModule):
    """Perfect Information Monte Carlo (PIMC) policy for Schafkopf.

    Samples possible opponent hand distributions and runs UCT (Monte Carlo
    Tree Search) rollouts on each sample, then aggregates results to select
    the best action.

    Parameters
    ----------
    samples : int
        Number of hand samples per decision.
    playouts : int
        Number of UCT playouts per sample.
    """

    def __init__(self, samples: int, playouts: int) -> None:
        super().__init__()
        self.samples: int = samples
        self.playouts: int = playouts
        self.rules: Rules = Rules()

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> int:
        allowed_actions, gamestate, player_cards = (
            batch["allowed_actions"],
            batch["game_state"],
            batch["current_player_cards"],
        )
        action, _ = self.run_mcts(gamestate, player_cards)

        index: int = -1
        if gamestate.game_stage == Rules.BIDDING:
            index =  self.rules.games.index(action)

        elif (
            gamestate.game_stage == Rules.CONTRA or gamestate.game_stage == Rules.RETOUR
        ):
            index = 9 + int(action)
        else:
            index = 11 + self.rules.cards.index(action)
        
        return index

    def _run_single_mcts(self, game_state: PublicGameState, player_cards: list[Card], rules: Rules, playouts: int) -> dict:
        """Run a single MCTS rollout with one sampled hand configuration.

        Parameters
        ----------
        game_state : PublicGameState
            Current public game state.
        player_cards : list[Card]
            The ego player's hand.
        rules : Rules
            Rules instance for action validation.
        playouts : int
            Number of playouts for this MCTS run.

        Returns
        -------
        dict
            Mapping from action to ``(visit_count, cumulative_rewards)``.
        """
        # Copy game_state if mutable to avoid cross-process contamination
        sampled_player_hands: list[list[Card]] = self.sample_player_hands(game_state, player_cards)
        mct = MonteCarloTree(game_state, sampled_player_hands, rules.allowed_actions(game_state, player_cards))
        mct.uct_search(playouts)
        return mct.get_action_count_rewards()

    def run_mcts(self, game_state: PublicGameState, player_cards: list[Card]) -> tuple[Any, float]:
        """Run PIMC search by aggregating multiple MCTS samples.

        Parameters
        ----------
        game_state : PublicGameState
            Current public game state.
        player_cards : list[Card]
            The ego player's hand.

        Returns
        -------
        tuple[Any, float]
            ``(best_action, confidence)`` where confidence is the fraction
            of visits going to the best action.
        """

        cummulative_action_count_rewards: dict = {}

        for i in range (self.samples):
            sampled_player_hands: list[list[Card]] = self.sample_player_hands(game_state, player_cards)
            mct: MonteCarloTree = MonteCarloTree(game_state,sampled_player_hands, self.rules.allowed_actions(game_state, player_cards))
            mct.uct_search(self.playouts)
            action_count_rewards: dict = mct.get_action_count_rewards()

            for action in action_count_rewards:
                if action in cummulative_action_count_rewards:
                    cummulative_action_count_rewards[action] = (cummulative_action_count_rewards[action][0] + action_count_rewards[action][0],
                                                            [cummulative_action_count_rewards[action][1][i] + action_count_rewards[action][1][i] for i in range(4)])
                else:
                    cummulative_action_count_rewards[action] = action_count_rewards[action]

        best_action: Any = max(cummulative_action_count_rewards.items(), key=lambda x : x[1][0])[0]
        visits: int = cummulative_action_count_rewards[best_action][0]
        return best_action, visits / sum([x[0] for x in cummulative_action_count_rewards.values()])

    def sample_player_hands(self, game_state: PublicGameState, ego_player_hand: list[Card]) -> list[list[Card]]:
        """Sample a valid card distribution for all players.

        Randomly distributes unseen cards among opponents and validates
        that the resulting distribution is consistent with the observed
        game history.

        Parameters
        ----------
        game_state : PublicGameState
            Current public game state.
        ego_player_hand : list[Card]
            The ego player's known hand.

        Returns
        -------
        list[list[Card]]
            Four player hands consistent with the game history.
        """

        # precomputations
        played_cards_set: set[Card] = set()
        for trick in game_state.course_of_game:
            for card in trick:
                if card != _NONE_CARD:
                    played_cards_set.add(card)
        ego_set: set[Card] = set(ego_player_hand)
        remaining_cards: list[Card] = [card for card in self.rules.cards if card not in played_cards_set and card not in ego_set]

        needed_player_cards: list[int] = [8, 8, 8, 8]

        for trick in range(game_state.trick_number + 1):
            for i, card in enumerate(game_state.course_of_game_playerwise[trick]):
                if card != _NONE_CARD:
                    needed_player_cards[i] -= 1

        needed_player_cards[game_state.current_player] = 0

        valid_card_distribution: bool = False
        player_cards: list[list[Card]] | None = None

        # loop over random card distributions until we found a valid one
        while not valid_card_distribution:

            # randomly distribute cards so that each player gets as many as he needs
            valid_card_distribution = True
            player_cards = [[], [], [], []]
            player_cards[game_state.current_player] = ego_player_hand
            random.shuffle(remaining_cards)

            from_card: int = 0
            for i, nededed_cards in enumerate(needed_player_cards):
                if i == game_state.current_player:
                    continue
                player_cards[i] = remaining_cards[from_card:from_card + nededed_cards]
                from_card += nededed_cards

            # check if with the current card distribution every made move was valid
            schafkopf_env: SchafkopfEnv = SchafkopfEnv()
            state, _, _ = schafkopf_env.set_state(PublicGameState(game_state.dealer), player_cards)

            while True:
                eval_game_state, allowed_actions = state["game_state"], state["allowed_actions"]

                if eval_game_state.game_stage == Rules.BIDDING:
                    action = eval_game_state.bidding_round[eval_game_state.current_player]
                    if action == None:
                        break
                    elif action not in allowed_actions:
                        valid_card_distribution = False
                        break
                elif eval_game_state.game_stage == Rules.CONTRA:
                    action = eval_game_state.contra[eval_game_state.current_player]
                    if action == None:
                        break
                    elif action not in allowed_actions:
                        valid_card_distribution = False
                        break
                elif eval_game_state.game_stage == Rules.RETOUR:
                    action = eval_game_state.retour[eval_game_state.current_player]
                    if action == None:
                        break
                    elif action not in allowed_actions:
                        valid_card_distribution = False
                        break
                else:
                    action = eval_game_state.course_of_game_playerwise[eval_game_state.trick_number][
                        eval_game_state.current_player]
                    if action == _NONE_CARD:
                        break
                    elif action not in allowed_actions:
                        valid_card_distribution = False
                        break
                state, _, _ = schafkopf_env.step(action)

        return player_cards