from typing import List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary

from .public_gamestate import PublicGameState
from .schafkopf_env import SchafkopfEnv
from .rules import Rules
from .utils import *
import numpy as np
import torch


class SchafkopfMultiAgentEnv(MultiAgentEnv):
    """
    This is a wrapper around the SchafkopfEnv class that allows it to be used as a MultiAgentEnv.
    """
    def __init__(self, config=None):
        super().__init__()
        self.agents = self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]
        self.env = SchafkopfEnv()
        
        self.MAX_ACTIONS = 44
        self.NUM_ACTIONS = 43
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32)
        self.action_history_len = 0


    # info_vector layout (38 dims, all ego-relative):
    #   game_stage:     11  (bidding, contra, retour, trick_0..trick_7)
    #   game_type:       7  (two-hot: 3 type bits + 4 color bits)
    #   game_player:     4  (one-hot, ego-relative)
    #   contra_retour:   8  (4 contra + 4 retour, ego-relative)
    #   first_player:    4  (one-hot, ego-relative)
    #   current_scores:  4  (divided by 120)
    INFO_VECTOR_SIZE = 38

    def get_observation_space(self, agent_id):
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

    def get_action_space(self, agent_id):
        return Discrete(self.NUM_ACTIONS)
    
    @property
    def num_agents(self):
        return 4

    def reset(self, *, seed=None, options=None):
        state, _ = self.env.reset(seed=seed)
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0
        # RLlib expects a dict of obs per agent
        return {self.agents[0]: self.state2obs(state)}, {self.agents[0]: state}

    def step(self, action_dict, *, seed=None, options=None):
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

    def state2obs(self, state):

        public_game_state = state["game_state"]
        player_cards = state["current_player_cards"]
        allowed_actions = state["allowed_actions"]

        observation = {}
        ego = public_game_state.current_player

        ############### player hand ##################
        observation["player_hand"] = one_hot_cards(player_cards).astype(np.int8)

        ############### info vector (38 dims, ego-relative) ##################
        # game_stage: 11 dims
        game_stage = np.zeros(11, dtype=np.float32)
        if public_game_state.game_stage == Rules.BIDDING:
            game_stage[0] = 1
        elif public_game_state.game_stage == Rules.CONTRA:
            game_stage[1] = 1
        elif public_game_state.game_stage == Rules.RETOUR:
            game_stage[2] = 1
        else:
            game_stage[3 + min(public_game_state.trick_number, 7)] = 1

        # game_type: 7 dims (two-hot)
        game_type_enc = two_hot_encode_game(public_game_state.game_type).astype(np.float32)

        # game_player: 4 dims (ego-relative)
        game_player_enc = np.zeros(4, dtype=np.float32)
        if public_game_state.game_player is not None:
            game_player_enc[(public_game_state.game_player - ego) % 4] = 1

        # contra_retour: 8 dims (ego-relative)
        contra_retour = np.zeros(8, dtype=np.float32)
        for p in range(4):
            if public_game_state.contra[p]:
                contra_retour[(p - ego) % 4] = 1
        for p in range(4):
            if public_game_state.retour[p]:
                contra_retour[4 + (p - ego) % 4] = 1

        # first_player: 4 dims (ego-relative)
        first_player_enc = np.zeros(4, dtype=np.float32)
        first_player_enc[(public_game_state.first_player - ego) % 4] = 1

        # current_scores: 4 dims (ego-relative, normalized)
        scores = np.zeros(4, dtype=np.float32)
        for p in range(4):
            scores[(p - ego) % 4] = public_game_state.scores[p] / 120.0

        observation["info_vector"] = np.concatenate(
            [game_stage, game_type_enc, game_player_enc, contra_retour, first_player_enc, scores]
        )

        ############### action history ##################
        observation["action_history"] = self.action_history.flatten()
        observation["action_history_len"] = int(self.action_history_len)

        ############### action mask ##################
        allowed_actions = self.env.rules.allowed_actions(public_game_state, player_cards)
        action_mask = np.zeros(43, dtype=np.int8)
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

    def render(self):
        self.env.render()

    def reset_with_fixed_cards(self, player_cards: List):
        '''
            Resets the env but with fixed card distributions. Useful for reading data transcripts.
        '''
        state, _, _ = self.env.set_state(PublicGameState(3), player_cards)
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0
        # RLlib expects a dict of obs per agent
        return {self.agents[0]: self.state2obs(state)}, {self.agents[0]: state}