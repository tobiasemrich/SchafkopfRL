# SchafkopfRL

Training a reinforcement learning agent to play the four player card game Schafkopfen. Uses an Actor-Critic Network and proximal policy optimization for training.

## Schafkopf Rules
Schafkopf is a traditional bavarian 4 player trick based card game with imperfect information. It has both competetive and cooperative game elements.

There are a lot of different variations (allowed game types, allowed doubling mechanisms, ...) and reward schemes. A good overview can be found at https://en.wikipedia.org/wiki/Schafkopf

The current focus of this project is to develop an AI that is able to play the basic game types Sauspiel, Farbsolo and Wenz (doubles like "Spritzen" and "Legen" will be added later) 

## Documentation

### Class Overview
Find the most imporatant classes for the training process below.

<img src="documentation/class_diagram.jpg">

### State and Action Space

The <b>state space </b> consists of three parts (necessary bits in brackets):

- info_vector (55)
  - game_type (7) [two bit encoding of color and type]
  - game_player (4)
  - first_player (4)
  - current_scores (4) [divided by 120 for normalization purpose]
  - remaining cards (32) [one hot encoded]
  - teams (4) [bits of players are set to 1, if Suchsau has been played already]
- game_history_sequence (x * 16)
    - course_of_game: x * (12 + 4) each played card in order plus the player that played it
- current_trick_sequence (y * 16)
    - current_trick: y * (12 + 4) each played card in order plus the player that played it

other players are encoded by position with respect to ego_player
The <b>action space</b> is a 41d vector that contains

- game type selection (9)
- card selection (32)

### Policy Network
<img src="documentation/network.jpg">

### Results
Playing against other players with 20/50 tariffs

<table>
    <tr>
        <th>Policy Network</th>
        <th>Hyperparameter</th>
        <th>against Random (cent/game)</th>
        <th>against Rule-based (cent/game)</th>
    </tr>
    <tr>
        <td>LSTM-based</td>
        <td>lr = 0.0001, batch_size = 50000, c1 = 0.5, c2 = 0.005, steps = 5M</td>
        <td>14.2</td>
        <td>9.7</td>
    </tr>
</table>

## Resources
- PPO Paper: https://arxiv.org/abs/1707.06347
- Pytorch implementation of PPO: https://github.com/nikhilbarhate99/PPO-PyTorch
- PPO parameter ranges: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
- Another Schafkopf RL project: https://github.com/clauszitzelsberger/Schafkopf_RL
- Another card game (Big2) tackled using RL with PPO: https://github.com/henrycharlesworth/big2_PPOalgorithm/
- Nice overview paper AI for card games: https://arxiv.org/pdf/1906.04439.pdf
