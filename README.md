# SchafkopfRL

Training a reinforcement learning agent to play the four player card game Schafkopfen. Uses an Actor-Critic Network and proximal policy optimization for training.

## Schafkopf Rules
Schafkopf is a traditional bavarian 4 player trick based card game with imperfect information. It has both competetive and cooperative game elements.

There are a lot of different variations (allowed game types, allowed doubling mechanisms, ...) and reward schemes. A good overview can be found at https://en.wikipedia.org/wiki/Schafkopf

In this project I will focus on the following rules:
- Long Cards (8 cards per player)
- Allowed Games: Sauspiel, Farbsolo, Wenz
- Tariffs: 20 for Sauspiel, 50 for Solo, 10 for Schneider/Schwarz or Laufende starting from 3 (from 2 for Wenz)
- Contra/Retour before first card was played
- Klopfen after first 4 cards

The current focus of this project is to develop an AI that is able to play the basic game types Sauspiel, Farbsolo and Wenz (doubles like "Spritzen" and "Legen" will be added later) 

## Documentation
### Basic Principle

1. The policy neural network (that decides what action to take at any given game state) is randomly initialized.
2. N games are played by 4 players using the current policy (N = 50K-100K)
3. A new policy is trained trying to make good decision more likely and bad decisions less likely using PPO
4. Replace the current policy by the new one and go back to 2.

<!--### Class Overview
Find the most imporatant classes for the training process below.

<img src="documentation/class_diagram.jpg">-->

### State and Action Space

The <b>state space </b> consists of three parts (necessary bits in brackets):

- info_vector (55)
  - game_type (7) [two bit encoding of color and type]
  - game_player (4)
  - first_player (4)
  - current_scores (4) [divided by 120 for normalization purpose]
  - remaining ego-player cards (32) [one hot encoded]
  - teams (4) [bits of players are set to 1, if Suchsau has been played already]
- game_history_sequence (x * 16)
    - course_of_game: x * (12 + 4) each played card in order plus the player that played it
- current_trick_sequence (y * 16)
    - current_trick: y * (12 + 4) each played card in order plus the player that played it

other players are encoded by position with respect to ego_player
The <b>action space</b> is a 41d vector that contains

- game type selection (9)
- card selection (32)

### LSTM-Based Policy Network
<img src="documentation/network.jpg">

### Results
Playing against other players with 20/50 tariffs (+10 for each "Laufenden" when more than 3):
- Random-Coward: Selects game randomly, but no solo game. Selects cards randomly.
- Rule-Based: Selects solo if enough trumps, otherwise non-solo game at random. Selects cards according to some simple human-imitating heuristics (play trump if player, don't play trump if non-player, play ace of color if possible, ...)

<table>
    <tr>
        <th>Policy Network</th>
        <th>Hyperparameter</th>
        <th>against Random-Coward(cent/game)</th>
        <th>against Rule-Based (cent/game)</th>
    </tr>
    <tr>
        <td>LSTM-based</td>
        <td>lr = 0.0001, update every 50K games, batch_size = 50K, c1 = 0.5, c2 = 0.005, steps = 5M</td>
        <td>14.2</td>
        <td>9.7</td>
    </tr>
    <tr>
        <td>Linear</td>
        <td>lr = 0.002, update every 100K games, batch_size = 600K, c1 = 0.5, c2 = 0.005, steps = 15M</td>
        <td>11.2</td>
        <td>8.5</td>
    </tr>
</table>

Example training run output of tensorboard (for the linear model)
<img src="documentation/example_run.png">

## Notes
### Version 28.04.2020
- Training takes a lot of time. After 15 days of continuous training the agent is still (slowly) improving.
- Large batchsize helps stabelizeing the training, but makes it slower. 
- Still have action shaping for the game selection: If cards are really good then solo is selected.
This was necessary in previous versions because the first thing the agent learns is not to play solos. With the large batchsize and some bugfixies this is probably not necessary anymore.
- Policy network has a lot of hidden units, should decrease in future versions.
- Playstyle:
  - Solos are played pretty good with small errors
     - The agent takes tricks if he does not have the played color
     - The agent playes trumps to pull trumps from other players
  - Sauspiele are not played so well but a lot of basic concepts are working
    - players take tricks if they do not have the played color
    - players play aces if possible
    - every player always wants to play. This maybe due to the reason that kontra is not implemented yet and playing on an Ace yields a higher probability of winning.
    - all players (including the game player) start playing colors and not trumps not sure why.
    - the team concept is not well understood: Agent sometimes plays higher trump than teammate. Agent does seldomly give points to certain trick of teammate.

## Resources
- PPO Paper: https://arxiv.org/abs/1707.06347
- Pytorch implementation of PPO: https://github.com/nikhilbarhate99/PPO-PyTorch
- PPO parameter ranges: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
- Another card game (Big2) tackled using RL with PPO: https://github.com/henrycharlesworth/big2_PPOalgorithm/
- Nice overview paper AI for card games: https://arxiv.org/pdf/1906.04439.pdf
