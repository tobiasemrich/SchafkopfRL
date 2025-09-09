from typing import List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary
from environment.schafkopf_env import SchafkopfEnv
from environment.rules import Rules
from environment.utils import *
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
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0


    def get_observation_space(self, agent_id):
        return Dict({
            "player_hand": MultiBinary(32),
            "action_history": Box(
                low=np.tile([-1, -1], (self.MAX_ACTIONS, 1)),
                high=np.tile([self.NUM_ACTIONS, 3], (self.MAX_ACTIONS, 1)),
                shape=(self.MAX_ACTIONS, 2),
                dtype=np.int32
            ),
            "action_history_len": Discrete(self.MAX_ACTIONS),
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

        ############### player hand ##################
        # array(32) that is 1 when card is held otherwise 0
        observation["player_hand"] = one_hot_cards(player_cards)

        ############### action history ##################
        observation["action_history"] = self.action_history
        observation["action_history_len"] = self.action_history_len

        ############### action mask ##################
        allowed_actions = self.env.rules.allowed_actions(public_game_state, player_cards)
        action_mask = np.zeros(43, dtype=np.int32)
        if public_game_state.game_stage == Rules.BIDDING:
            action_mask[0:9] = one_hot_games(allowed_actions)
        elif public_game_state.game_stage == Rules.CONTRA or public_game_state.game_stage == Rules.RETOUR:
            action_mask[9] = 1
            if np.any(allowed_actions):
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
        state, _ = self.env.set_state(PublicGameState(3), player_cards)
        self.action_history = np.full((self.MAX_ACTIONS, 2), -1, dtype=np.int32) # represents array of (action, player_id) tuples
        self.action_history_len = 0
        # RLlib expects a dict of obs per agent
        return {self.agents[0]: self.state2obs(state)}, {self.agents[0]: state}