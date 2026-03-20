from typing import Any, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary

from .public_gamestate import PublicGameState
from .schafkopf_env import SchafkopfEnv
from .rules import Rules
from .utils import *
import numpy as np
import torch


class SchafkopfMultiAgentEnv(MultiAgentEnv):
    """RLlib multi-agent wrapper around the Schafkopf card game environment.

    Translates between the discrete action/observation interface expected by
    RLlib and the internal ``SchafkopfEnv`` representation. Each of the four
    players is treated as a separate agent that acts in turn.

    Parameters
    ----------
    config : dict, optional
        Environment configuration (currently unused).
    """
    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.agents: list[str] = ["player_0", "player_1", "player_2", "player_3"]
        self.possible_agents: list[str] = self.agents
        self.env: SchafkopfEnv = SchafkopfEnv()
        
        self.MAX_ACTIONS: int = 44
        self.NUM_ACTIONS: int = 43
        self.action_history: np.ndarray = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32)
        self.action_history_len: int = 0


    # info_vector layout (38 dims, all ego-relative):
    #   game_stage:     11  (bidding, contra, retour, trick_0..trick_7)
    #   game_type:       7  (two-hot: 3 type bits + 4 color bits)
    #   game_player:     4  (one-hot, ego-relative)
    #   contra_retour:   8  (4 contra + 4 retour, ego-relative)
    #   first_player:    4  (one-hot, ego-relative)
    #   current_scores:  4  (divided by 120)
    INFO_VECTOR_SIZE: int = 38

    @property
    def observation_space(self) -> Dict:
        """Return the observation space shared by all agents.

        Returns
        -------
        Dict
            Gymnasium Dict space with keys ``player_hand``, ``info_vector``,
            ``action_history``, ``action_history_len``, and ``action_mask``.
        """
        return Dict({
            "player_hand": MultiBinary(32),
            "info_vector": Box(low=0.0, high=1.0, shape=(self.INFO_VECTOR_SIZE,), dtype=np.float32),
            "action_history": Box(
                low=-1,
                high=self.NUM_ACTIONS,
                shape=(self.MAX_ACTIONS * 2,),
                dtype=np.int32
            ),
            "action_history_len": Discrete(self.MAX_ACTIONS + 1),
            "action_mask": MultiBinary(self.NUM_ACTIONS)
        })

    @property
    def action_space(self) -> Discrete:
        """Return the discrete action space shared by all agents.

        Returns
        -------
        Discrete
            Gymnasium Discrete space with 43 actions.
        """
        return Discrete(self.NUM_ACTIONS)
    
    @property
    def num_agents(self):
        return 4

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, dict], dict[str, dict]]:
        """Reset the environment and deal new cards.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        tuple[dict[str, dict], dict[str, dict]]
            ``(observations, infos)`` keyed by the first acting agent.
        """
        state, _ = self.env.reset(seed=seed)
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0
        # RLlib expects a dict of obs per agent
        return {self.agents[0]: self.state2obs(state)}, {self.agents[0]: state}

    def step(self, action_dict: dict[str, Any], *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one action in the environment.

        Exactly one agent acts per step. The action is translated from
        the discrete NN output to the internal game representation.

        Parameters
        ----------
        action_dict : dict[str, Any]
            Mapping from agent id to the chosen discrete action.
        seed : int, optional
            Random seed (unused).
        options : dict, optional
            Additional options (unused).

        Returns
        -------
        tuple[dict, dict, dict, dict, dict]
            ``(observations, rewards, terminateds, truncateds, infos)``
            each keyed by agent id.
        """
        # RLlib provides a dict of actions per agent, but only one agent acts at a time in Schafkopf
        # Find the acting agent (should be only one key in action_dict)
        assert len(action_dict) == 1, "Only one agent acts at a time."
        agent_id, action = next(iter(action_dict.items()))

        if isinstance(action, torch.Tensor):
            action = action.item()

        # translate action output form NN back to base env
        env_action = None
        if action <= 8: #self.env.public_gamestate.game_stage == Rules.BIDDING:
            env_action =  self.env.rules.games[action]
        elif action <= 10: #self.env.public_gamestate.game_stage == Rules.CONTRA or self.env.public_gamestate.game_stage == Rules.RETOUR:
            env_action =   bool(action - 9)
        else: #trick stage
            env_action =   self.env.rules.cards[action - 11]
        
        if env_action not in self.env.last_allowed_actions:
            print(f"Invalid action encountered, hopefully we are in precheck, will sample a valid one")
            env_action = self.env.last_allowed_actions[0]


        state, rew, done = self.env.step(env_action)

        # safe the action in the action history
        self.action_history[self.action_history_len] = [action, int(agent_id[-1])]
        self.action_history_len += 1

        # RLlib expects a dict of rewards, obs, and done flags per agent
        current_player = self.agents[self.env.public_gamestate.current_player]
        observations = {current_player: self.state2obs(state)}
        rewards = {p: rew[i] for i, p in enumerate(self.agents)}
        terminateds = {p: done for p in self.agents}
        terminateds["__all__"] = done
        truncated = {p: False for p in self.agents}
        truncated["__all__"] = False
        info_dict = {current_player: state} # return the raw state from SchafkopfEnv for the rule based policies
        return observations, rewards, terminateds, truncated, info_dict

    def state2obs(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert the internal game state to an ego-relative observation dict.

        Encodes the player's hand, game info vector, action history, and
        action mask into the format expected by the observation space.

        Parameters
        ----------
        state : dict[str, Any]
            Raw state dict from ``SchafkopfEnv``.

        Returns
        -------
        dict[str, Any]
            Observation matching ``observation_space``.
        """

        public_game_state: PublicGameState = state["game_state"]
        player_cards: list = state["current_player_cards"]
        allowed_actions: list = state["allowed_actions"]

        observation: dict[str, Any] = {}
        ego: int = public_game_state.current_player

        ############### player hand ##################
        observation["player_hand"] = one_hot_cards(player_cards).astype(np.int8)

        ############### info vector (38 dims, ego-relative) ##################
        # game_stage: 11 dims
        game_stage: np.ndarray = np.zeros(11, dtype=np.float32)
        if public_game_state.game_stage == Rules.BIDDING:
            game_stage[0] = 1
        elif public_game_state.game_stage == Rules.CONTRA:
            game_stage[1] = 1
        elif public_game_state.game_stage == Rules.RETOUR:
            game_stage[2] = 1
        else:
            game_stage[3 + min(public_game_state.trick_number, 7)] = 1

        # game_type: 7 dims (two-hot)
        game_type_enc: np.ndarray = two_hot_encode_game(public_game_state.game_type).astype(np.float32)

        # game_player: 4 dims (ego-relative)
        game_player_enc: np.ndarray = np.zeros(4, dtype=np.float32)
        if public_game_state.game_player is not None:
            game_player_enc[(public_game_state.game_player - ego) % 4] = 1

        # contra_retour: 8 dims (ego-relative)
        contra_retour: np.ndarray = np.zeros(8, dtype=np.float32)
        for p in range(4):
            if public_game_state.contra[p]:
                contra_retour[(p - ego) % 4] = 1
        for p in range(4):
            if public_game_state.retour[p]:
                contra_retour[4 + (p - ego) % 4] = 1

        # first_player: 4 dims (ego-relative)
        first_player_enc: np.ndarray = np.zeros(4, dtype=np.float32)
        first_player_enc[(public_game_state.first_player - ego) % 4] = 1

        # current_scores: 4 dims (ego-relative, normalized)
        scores: np.ndarray = np.zeros(4, dtype=np.float32)
        for p in range(4):
            scores[(p - ego) % 4] = public_game_state.scores[p] / 120.0

        observation["info_vector"] = np.concatenate(
            [game_stage, game_type_enc, game_player_enc, contra_retour, first_player_enc, scores]
        )

        ############### action history ##################
        ego_history = self.action_history.copy()
        mask = ego_history[:, 1] != -1
        ego_history[mask, 1] = (ego_history[mask, 1] - ego)%4
        observation["action_history"] = self.action_history.flatten()
        observation["action_history_len"] = int(self.action_history_len)

        ############### action mask ##################
        allowed_actions = self.env.rules.allowed_actions(public_game_state, player_cards)
        action_mask: np.ndarray = np.zeros(43, dtype=np.int8)
        if public_game_state.game_stage == Rules.BIDDING:
            action_mask[0:9] = one_hot_games(allowed_actions)
        elif public_game_state.game_stage == Rules.CONTRA or public_game_state.game_stage == Rules.RETOUR:
            action_mask[9] = 1
            if any(allowed_actions):
                action_mask[10] = 1
        else:
            action_mask[11:] = one_hot_cards(allowed_actions)

        observation["action_mask"] = action_mask

        return observation

    def render(self) -> None:
        self.env.render()

    def reset_with_fixed_cards(self, player_cards: List) -> tuple[dict[str, dict], dict[str, dict]]:
        """Reset the environment with a predetermined card distribution.

        Useful for replaying game transcripts or deterministic testing.

        Parameters
        ----------
        player_cards : list
            List of four hands, one per player.

        Returns
        -------
        tuple[dict[str, dict], dict[str, dict]]
            ``(observations, infos)`` keyed by the first acting agent.
        """
        state, _, _ = self.env.set_state(PublicGameState(3), player_cards)
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0
        # RLlib expects a dict of obs per agent
        return {self.agents[0]: self.state2obs(state)}, {self.agents[0]: state}