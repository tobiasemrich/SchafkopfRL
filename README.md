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
     - A sample game [number in brackets are the selection probabilities of the chosen action]
<pre>
Bidding Round: (0)gras sauspiel [0.912]  (1*)eichel solo [1.000]  (2)eichel sauspiel [0.951]  (3)schellen sauspiel [0.011]  

Played Game: eichel solo played by player: 1

Course of game
(0)schellen unter[0.643]      (1^)eichel unter[0.364]       (2)eichel zehner[1.000]       (3*)herz ober[1.000]          
(0)herz neuner[0.058]         (1*)schellen koenig[1.000]    (2)schellen siebener[0.980]   (3^)schellen achter[0.104]    
(0)gras unter[1.000]          (1^)gras ober[0.779]          (2*)eichel ober[0.524]        (3)eichel neuner[1.000]       
(0)herz koenig[0.604]         (1*)eichel achter[0.440]      (2^)herz sau[0.149]           (3)herz achter[1.000]         
(0)gras siebener[0.343]       (1^*)schellen ober[1.000]     (2)herz unter[1.000]          (3)schellen zehner[0.294]     
(0)herz zehner[0.035]         (1^*)eichel siebener[0.957]   (2)gras sau[0.703]            (3)gras zehner[0.832]         
(0)gras neuner[0.458]         (1^*)eichel koenig[0.285]     (2)gras achter[0.971]         (3)gras koenig[0.621]         
(0)herz siebener[1.000]       (1^*)eichel sau[1.000]        (2)schellen sau[1.000]        (3)schellen neuner[1.000]     

Scores: [0, 95, 8, 17]

Rewards: [-60, 180, -60, -60]
 </pre>
 
  - Sauspiele are not played so well but a lot of basic concepts are working
    - players take tricks if they do not have the played color
    - players play aces if possible
    - every player always wants to play. This maybe due to the reason that kontra is not implemented yet and playing on an Ace yields a higher probability of winning.
    - all players (including the game player) start playing colors and not trumps not sure why.
    - the team concept is not well understood: Agent sometimes plays higher trump than teammate. Agent does seldomly give points to certain trick of teammate.
    - a sample game [number in brackets are the selection probabilities of the chosen action]
<pre>
Bidding Round: (0)gras sauspiel [0.564]  (1)gras sauspiel [0.521]  (2)schellen sauspiel [0.590]  (3*)eichel sauspiel [0.997]  

Played Game: eichel sauspiel played by player: 3

Course of game
(0)herz siebener[0.839]       (1*)gras unter[0.653]         (2)herz unter[0.570]          (3^)herz neuner[0.166]        
(0)schellen siebener[0.714]   (1^)schellen achter[0.377]    (2)schellen zehner[1.000]     (3*)herz koenig[0.034]        
(0)herz zehner[1.000]         (1*)eichel ober[0.844]        (2)schellen unter[0.801]      (3^)eichel unter[0.467]       
(0*)eichel sau[1.000]         (1^)eichel neuner[0.965]      (2)eichel siebener[1.000]     (3)eichel achter[1.000]       
(0^)eichel zehner[0.210]      (1)schellen ober[0.993]       (2)gras achter[0.140]         (3*)herz ober[0.966]          
(0)eichel koenig[0.561]       (1)schellen koenig[0.542]     (2*)gras ober[1.000]          (3^)herz achter[0.925]        
(0)gras koenig[1.000]         (1)gras zehner[1.000]         (2^)gras neuner[0.527]        (3*)gras sau[1.000]           
(0)schellen sau[1.000]        (1)schellen neuner[1.000]     (2)gras siebener[1.000]       (3^*)herz sau[1.000]          

Scores: [11, 21, 11, 77]

Rewards: [20, -20, -20, 20]
</pre>
    
    

## Resources
- PPO Paper: https://arxiv.org/abs/1707.06347
- Pytorch implementation of PPO: https://github.com/nikhilbarhate99/PPO-PyTorch
- PPO parameter ranges: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
- Another card game (Big2) tackled using RL with PPO: https://github.com/henrycharlesworth/big2_PPOalgorithm/
- Nice overview paper AI for card games: https://arxiv.org/pdf/1906.04439.pdf
