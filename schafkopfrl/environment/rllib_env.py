from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Box, Discrete, MultiBinary
from .schafkopf_env import SchafkopfEnv
import numpy as np
from rules import Rules
from utils import *
import torch

class SchafkopfMultiAgentEnv(MultiAgentEnv):
    """
    This is a wrapper around the SchafkopfEnv class that allows it to be used as a MultiAgentEnv.
    """
    def __init__(self, seed=None):
        super().__init__()
        self.env = SchafkopfEnv(seed=seed)
        self.observation_space = Dict({
            "info_vector": MultiBinary(70),
            "game_history": Box(shape=(32, 16), dtype=np.int32),
            "hand": MultiBinary(32),
            "action_mask": MultiBinary(43)
        })

        self.action_space = Discrete(32)  # Placeholder
        self.num_agents = 4
        self.agent_ids = ["player_" + str(i) for i in range(self.num_agents)]

    def reset(self):
        state, _ = self.env.reset(seed=None)
        # RLlib expects a dict of obs per agent
        return {"player_0": self.state2obs(state)}

    def step(self, action_dict):
        # RLlib provides a dict of actions per agent, but only one agent acts at a time in Schafkopf
        # Find the acting agent (should be only one key in action_dict)
        assert len(action_dict) == 1, "Only one agent acts at a time."
        agent_id, action = next(iter(action_dict.items()))
        state, rewards = self.env.step(action)
        # Check if game is done
        done = self.env.public_gamestate.trick_number == 8
        # RLlib expects a dict of rewards, obs, and done flags per agent
        current_player = "player_" + str(self.env.public_gamestate.current_player)
        obs_dict = {current_player: self.state2obs(state)}
        reward_dict = {str(i): rewards[i] for i in range(self.num_agents)}
        done_dict = {str(i): done for i in range(self.num_agents)}
        done_dict["__all__"] = done
        info_dict = {str(i): {} for i in range(self.num_agents)}
        return obs_dict, reward_dict, done_dict, info_dict

    def state2obs(self, state):
        """
        Convert the state to an observation.
        observation_space:
        - info_vector: 70 (74)
            - game_stage: 11
            - game_type: 7 [two bit encoding]
            - game_player: 4
            - contra_retour: 8
            - first_player: 4
            - current_scores: 4 (divided by 120 for normalization purpose)
            - player cards: 32
            - ( teams: 4 [bits of players are set to 1])
            - (should add remaining cards)
        - game_history: x * 16
            - course_of_game: x * (12 + 4) each played card in order plus the player that played it

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
        else:
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

        #course of game
        course_of_game_enc = np.zeros((1, 16))
        for trick in range(len(game_state.course_of_game)):
            for card in range(len(game_state.course_of_game[trick])):
                if game_state.course_of_game[trick][card] == [None, None]:
                    continue
                else:
                    card_player = game_state.first_player
                    if trick != 0:
                        card_player = game_state.trick_owner[trick - 1]
                    card_player = (card_player + card) % 4
                    card_player_enc = np.zeros(4)
                    card_player_enc[(card_player-ego_player)%4] = 1
                    course_of_game_enc = np.vstack((course_of_game_enc, np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))


        info_vector = np.concatenate((game_stage, game_enc, game_player_enc, contra_retour, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player_cards)))

        if course_of_game_enc.shape[0] > 1:
            course_of_game_enc = np.delete(course_of_game_enc, 0, 0)
        course_of_game_enc = torch.tensor(course_of_game_enc).float()
        course_of_game_enc = course_of_game_enc.view(len(course_of_game_enc),1,  16)

        ############### allowed actions ##################
        allowed_actions_enc = np.zeros(43)
        if game_state.game_stage == Rules.BIDDING:
            allowed_actions_enc[0:9] = one_hot_games(allowed_actions)
        elif game_state.game_stage == Rules.CONTRA or game_state.game_stage == Rules.RETOUR:
            allowed_actions_enc[10] = 1
            if np.any(allowed_actions):
                allowed_actions_enc[9] = 1
        else:
            allowed_actions_enc[11:] = one_hot_cards(allowed_actions)


        return {
            "info_vector": torch.tensor(info_vector).float(),
            "game_history": course_of_game_enc,
            "action_mask": torch.tensor(allowed_actions_enc).float()
        }