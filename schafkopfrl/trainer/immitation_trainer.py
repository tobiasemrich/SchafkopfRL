import torch
from sqlitedict import SqliteDict

from public_gamestate import PublicGameState
from schafkopf_env import SchafkopfEnv
from settings import Settings
from tensorboard import program

def main():

  #load database
  games = SqliteDict('../sauspiel/games.sqlite')
  for g in games:

    if g == "100001917":
      print(g)

    game = games[g]
    if len(game.sonderregeln) == 0:
      states, actions = get_states_actions(game)
      print(g)

def get_states_actions(game_transcript):

  states = []
  actions = []

  schafkopf_env = SchafkopfEnv()

  state, _, _ = schafkopf_env.set_state(PublicGameState(3), [game_transcript.player_hands[i] for i in range(4)])
  states.append(state)

  #Bidding stage

  game_player = None
  game_type = None

  if len(game_transcript.bidding_round) != 4:  # not all said weiter
    player_bidding = None
    for i in range(1, 5):
      if "Vortritt" not in game_transcript.bidding_round[-i]:
        player_bidding = game_transcript.bidding_round[-i]
        break

    if player_bidding.startswith("Ex-Sauspieler"):
      game_player = game_transcript.player_dict[player_bidding.split(" ")[0] + " " + player_bidding.split(" ")[1]]
    else:
      game_player = game_transcript.player_dict[player_bidding.split(" ")[0]]

    if "Hundsgfickte" in player_bidding:
      game_type = [0, 0]
    elif "Blaue" in player_bidding:
      game_type = [2, 0]
    elif "Alte" in player_bidding:
      game_type = [3, 0]
    elif "Schelle" in player_bidding:
      game_type = [0, 2]
    elif "Herz" in player_bidding:
      game_type = [1, 2]
    elif "Gras" in player_bidding:
      game_type = [2, 2]
    elif "Eichel" in player_bidding:
      game_type = [3, 2]
    elif "Wenz" in player_bidding:
      game_type = [None, 1]


  for i in range(4):
    action = [None, None]
    if i == game_player:
      action = game_type
    actions.append(action)
    state, _, _ = schafkopf_env.step(action)
    states.append(state)

  if len(game_transcript.bidding_round) != 4: # if not all said weiter

    con_ret = [game_transcript.player_dict[p] for p in game_transcript.kontra]

    #CONTRA stage
    for i in range(4):
      action = False
      if len(con_ret) > 0 and i == con_ret[0]:
        action = True
      actions.append(action)
      state, _, _ = schafkopf_env.step(action)
      states.append(state)

    # RETOUR stage
    if len(con_ret) > 0:
      for i in range(4):
        action = False
        if len(con_ret) == 2 and i == con_ret[1]:
          action = True
        actions.append(action)
        state, _, _ = schafkopf_env.step(action)
        states.append(state)

    # TRICK stage

    for trick in range(8):
      for c in range(4):
        played_card = game_transcript.course_of_game[trick][c]
        actions.append(played_card)
        state, _, _ = schafkopf_env.step(played_card)
        states.append(state)

  states.pop()#last state is the final state, so nothing to learn from here
  return states, actions

if __name__ == '__main__':
  main()