# SchafkopfRL

Developing an AI agent for play the bavarian four-player card game Schafkopfen. The main components of this repo are:
- <b>Schafkopf Environment</b>: A multi-agent environment that allows agents to play Schafkopfen. See Schafkopf Rules for the supported rule set.
- <b>Agents</b>: A set of AI agents that are able to play with different degrees of strength
  - PPO Agent: RL Agent trained with proximal policy optimization.
    - Linear: Using a 1D vector state representation of the current game state and an Actor-Critic Network that has a linear input layer.
    - LSTM: Using a complex state representation (e.g., representing played cards as sequences) and an Actor-Critic Network that also hast LSTM input layers.
  - PIMC Agents: Using Monte-Carlo-Tree Search for this imperfect information game. Samples opponent hands several times and performs MCTS on each instance.
    - Vanilla: Random sampling of opponent hands
    - Hand-Predictor: utilizes a neural network for predicting card distribution amongst opponents. Trained by self-play.
  - Simple Agents: Agents with simple hard-coded rules
    - Random: performs each action random (only valid actions)
    - Random-Coward: performs each action random, but never plays a solo and never doubles the game.
    - Rule-based: Plays solo if enough trumps, otherwise non-solo game at random. Selects cards according to some simple human-imitating heuristics (play trump if player, don't play trump if non-player, play ace of color if possible, ...)
- <b>Trainer:</b>  Trainer class for training the model based-players


## Schafkopf Rules
Schafkopf is a traditional bavarian 4 player trick based card game with imperfect information. It has both competetive and cooperative game elements.

There are a lot of different variations (allowed game types, allowed doubling mechanisms, ...) and reward schemes. A good overview can be found at https://en.wikipedia.org/wiki/Schafkopf

In this project I will focus on the following rules:
- Long Cards (8 cards per player)
- Allowed Games: Sauspiel, Farbsolo, Wenz
- Tariffs: 20 for Sauspiel, 50 for Solo, 10 for Schneider/Schwarz or Laufende starting from 3 (from 2 for Wenz)
- Contra/Retour before first card was played

## Current Results
These results are just preliminary and subject to change. The shown numbers are cents/game

<table>
    <tr><th></th><th>HP PIMC(10, 40)</th><th>PIMC(10, 40)</th><th>PPO (lstm)</th><th>PPO (linear)</th><th>rule-based</th><th>random-coward</th><th>random</th></tr>
    <tr><td>HP PIMC(10, 40)</td>-<td>4.9</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>PIMC(10, 40)</td><td>- 4.9</td>-<td></td>~ 8.0<td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>PPO (lstm)</td><td></td><td>~ - 8.0</td><td> - </td><td></td><td>9.7</td><td>14.2</td><td></td></tr>
    <tr><td>PPO (linear)</td><td></td><td></td><td></td><td></td><td>8.5</td><td>11.2</td><td></td></tr>
</table>

## PPO Agent
### Basic Principle

1. The policy neural network (that decides what action to take at any given game state) is randomly initialized.
2. N games are played by 4 players using the current policy (N = 50K-100K)
3. A new policy is trained trying to make good decision more likely and bad decisions less likely using PPO
4. Replace the current policy by the new one and go back to 2.


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

### Example Run
Hyperparameters used: lr = 0.002, update every 100K games, batch_size = 600K, c1 = 0.5, c2 = 0.005, steps = 15M

Hyperparameters used for LSTM: lr = 0.0001, update every 50K games, batch_size = 50K, c1 = 0.5, c2 = 0.005, steps = 5M

Example training run output of tensorboard (for the linear model)
<img src="documentation/example_run.png">

## PIMC Agent (Perfect Information Monte Carlo Agent)
The basic principle of the PIMC(n, m) Agent is to do n times:
   1. distribute remaining cards to opponents
   2. perform Monte-Carlo Tree Search (MCTS) m times with some agent (usually random but possibility to use other probabilistic agents)
    
Eventually, take action with the highest cummulative visits over the n runs

Hand-Prediction PIMC Agent learns an NN to estimate the distribution of remaining cards amongst opponents to improve Step 1.

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

### Version 02.10.2020
 - Added Contra and Retour
 - Added PIMC player (in particular Perfect Infromation Monte Carlo)
 - PIMC Player performs unfortunately much better than expected. Tournament with 4 players for 1000 games resulted in the following per game rewards
    <table>
        <tr><td>PIMC_Player(5, 20)</td><td>-9.24</td></tr>
        <tr><td>PIMC_Player(10, 40)</td><td>12.6</td></tr>
        <tr><td>PIMC_Player(10, 100)</td><td>13.78</td></tr>
        <tr><td>RLPLayer</td><td>-17.14</td></tr>
    </table>
    
 - Problems of PIMC player (good article: https://core.ac.uk/download/pdf/30267707.pdf)
    - non-locality: "Non-locality is an issue that arises since history can matter in a hidden information game". Non-locality shows very clearly when an MCTS player is playing against another player X who chose to play a solo game. The MCTS player will then sample possible card distibutions and determine that this player X will often loose his solo game. Thus the MCTS player will usually double (contra) the game when someone plays a solo.
    - strategy-fusion: could not find a good example for this in schafkopf so far.

 - Ideas to improve PIMC player:
   - icorporate the probability of a card distribution (probability of the players playing the cards they have played given the hand they have)
   
### Version 04.11.2020
- Added a hand prediction network to PIMC (HP_MCTS_Player)
  - Input: info_vector + Sequence of played cards
  - Network: 1) Linear Layer + LSTM Layer 2) 2 x Linear Layer 3) 32x4 tensor
  - Output: probability for each card to be at each players hand
- Trained by iteratively playing n = 400 games and then updating. Playing a game is really slow (10 secs / game)
- MCTS_Player(10, 40) vs. Smart_MCTS_Player(10, 40) = -4.9 vs 4.9 over 3K games, so this really improves the PIMC player. Still not close to human level IMHO.

## Resources
- PPO Paper: https://arxiv.org/abs/1707.06347
- Pytorch implementation of PPO: https://github.com/nikhilbarhate99/PPO-PyTorch
- PPO parameter ranges: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
- Another card game (Big2) tackled using RL with PPO: https://github.com/henrycharlesworth/big2_PPOalgorithm/
- Nice overview paper AI for card games: https://arxiv.org/pdf/1906.04439.pdf
- MCTS for imperfect information games https://core.ac.uk/download/pdf/30267707.pdf
- DL model for predicting opponent hands for PIMC https://www.aaai.org/ojs/index.php/AAAI/article/view/3909/3787
