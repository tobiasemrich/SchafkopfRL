from gymnasium import RewardWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Discrete, Dict
from environment.schafkopf_env import SchafkopfEnv
from environment.rules import Rules
from environment.utils import *
import numpy as np

class LinearSchafkopfMultiAgentEnv(MultiAgentEnv):
    """
    This is a wrapper around the SchafkopfEnv class that allows it to be used as a MultiAgentEnv.
    """
    def __init__(self, config=None, seed=None):
        super().__init__()
        self.agents = self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]
        self.env = SchafkopfEnv(seed=seed)
        #self.action_space = self.get_action_space("player_0") # only for fallback
        #self.observation_space = self.get_observation_space("player_0") # only for fallback


    def get_observation_space(self, agent_id):
        return Dict({
            "observation": Box(low=0, high=1, shape=(102,), dtype=np.float32),
            "action_mask": Box(0, 1, shape=(43,), dtype=np.float32)
        })

    def get_action_space(self, agent_id):
        return Discrete(43)
    
    @property
    def num_agents(self):
        return 4

    def reset(self, *, seed=None, options=None):
        state, _ = self.env.reset(seed=None)
        # RLlib expects a dict of obs per agent
        info_dict = {i : {} for i in self.agents}
        return {self.agents[0]: self.state2obs(state)}, info_dict

    def step(self, action_dict):
        # RLlib provides a dict of actions per agent, but only one agent acts at a time in Schafkopf
        # Find the acting agent (should be only one key in action_dict)
        assert len(action_dict) == 1, "Only one agent acts at a time."
        agent_id, action = next(iter(action_dict.items()))


        # translate action output form NN back to base env
        env_action = None
        if action <= 8: #self.env.public_gamestate.game_stage == Rules.BIDDING:
            env_action =  self.env.rules.games[action]
        elif action <= 10: #self.env.public_gamestate.game_stage == Rules.CONTRA or self.env.public_gamestate.game_stage == Rules.RETOUR:
            env_action =   action.item() - 9 == 0
        else: #trick stage
            env_action =   self.env.rules.cards[action - 11]
        
        if env_action not in self.env.last_allowed_actions:
            print(f"Invalid action encountered, hopefully we are in precheck, will sample a valid one")
            env_action = self.env.last_allowed_actions[0]


        state, rew, done = self.env.step(env_action)
        # RLlib expects a dict of rewards, obs, and done flags per agent
        current_player = self.agents[self.env.public_gamestate.current_player]
        observations = {current_player: self.state2obs(state)}
        rewards = {p: rew[i] for i, p in enumerate(self.agents)}
        terminateds = {p: done for p in self.agents}
        terminateds = {"__all__": done}
        truncated = {p: False for p in self.agents}
        truncated["__all__"] = False
        info_dict = {i : {} for i in self.agents}
        return observations, rewards, terminateds, truncated, info_dict

    def state2obs(self, state):
        """
        Convert the state to an observation.
        observation_space:
        - info_vector: 102
            - game_stage: 11
            - game_type: 7 [two bit encoding]
            - game_player: 4
            - contra_retour: 8
            - first_player: 4
            - current_scores: 4 (divided by 120 for normalization purpose)
            - players cards: 32
            - remaining cards: 32
           

        action_size (43):
            - games: 9
            - contra/double: 2
            - cards:  32
        """
        game_state = state["game_state"]
        player_cards = state["current_player_cards"]
        allowed_actions = state["allowed_actions"]

        ############### gamestate ##################
        ego_player = game_state.current_player

        #game stage
        game_stage = np.zeros(11)
        if game_state.game_stage == Rules.BIDDING:
            game_stage[0] = 1
        elif game_state.game_stage == Rules.CONTRA:
            game_stage[1] = 1
        elif game_state.game_stage == Rules.RETOUR:
            game_stage[2] = 1
        elif game_state.trick_number != 8:
            game_stage[3+game_state.trick_number] = 1


        game_enc = two_hot_encode_game(game_state.game_type)

        game_player_enc = np.zeros(4)
        if game_state.game_player != None:
            game_player_enc[(game_state.game_player-ego_player)%4] = 1

        contra_retour = np.zeros(8)
        for p in range (4):
            if game_state.contra[p]:
                contra_retour[(p-ego_player)%4] = 1
        for p in range (4):
            if game_state.retour[p]:
                contra_retour[4 + (p-ego_player)%4] = 1

        first_player_enc = np.zeros(4)
        first_player_enc[(game_state.first_player-ego_player)%4] = 1

        remaining_cards = one_hot_cards(card for trick in game_state.course_of_game for card in trick if card[0] != None)


        info_vector = np.concatenate((game_stage, game_enc, game_player_enc, contra_retour, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player_cards), remaining_cards))

        ############### allowed actions ##################
        allowed_actions_enc = np.zeros(43, dtype=np.float32)
        if game_state.game_stage == Rules.BIDDING:
            allowed_actions_enc[0:9] = one_hot_games(allowed_actions)
        elif game_state.game_stage == Rules.CONTRA or game_state.game_stage == Rules.RETOUR:
            allowed_actions_enc[10] = 1
            if np.any(allowed_actions):
                allowed_actions_enc[9] = 1
        else:
            allowed_actions_enc[11:] = one_hot_cards(allowed_actions)

        return {
            "observation": np.array(info_vector, dtype=np.float32),
            "action_mask": allowed_actions_enc
        }