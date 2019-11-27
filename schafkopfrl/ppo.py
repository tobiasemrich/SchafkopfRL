import time
from os import listdir

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import  StepLR

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from schafkopfrl.experience_dataset import ExperienceDataset, custom_collate
from schafkopfrl.game_simulation import Game_Simulation

from schafkopfrl.models.actor_critic4 import ActorCriticNetwork4
from schafkopfrl.models.actor_critic5 import ActorCriticNetwork5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, policy, lr_params, betas, gamma, K_epochs, eps_clip, batch_size, c1=0.5, c2=0.01):
        [self.lr, self.lr_stepsize, self.lr_gamma] = lr_params

        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.batch_size= batch_size

        self.c1 = c1
        self.c2 = c2

        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.lr, betas=betas, weight_decay=1e-5)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.lr_stepsize, gamma=self.lr_gamma)
        self.policy_old = type(policy)().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.writer = SummaryWriter()

    def update(self, memory, i_episode):

        #alpha = 1-i_episode/900000
        alpha = 1
        self.policy.train()

        #decay learning rate and eps_clip
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr * alpha
        adapted_eps_clip = self.eps_clip*alpha

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)

        # Create dataset from collected experiences
        experience_dataset = ExperienceDataset(memory.states, memory.actions, memory.allowed_actions, memory.logprobs, rewards)
        training_generator = data.DataLoader(experience_dataset, collate_fn=custom_collate, batch_size=self.batch_size)


        # Optimize policy for K epochs:
        avg_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        count = 0
        for _ in range(self.K_epochs):
            for old_states, old_actions, old_allowed_actions, old_logprobs, old_rewards in training_generator:

                # Transfer to GPU
                old_states = [old_state.to(device) for old_state in old_states]
                old_actions, old_allowed_actions, old_logprobs, old_rewards = old_actions.to(device), old_allowed_actions.to(device), old_logprobs.to(device), old_rewards.to(device)

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states,old_allowed_actions, old_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = old_rewards.detach() - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - adapted_eps_clip, 1 + adapted_eps_clip) * advantages
                value_loss = self.MseLoss(state_values, old_rewards)
                loss = -torch.min(surr1, surr2) + self.c1 * value_loss - self.c2 * dist_entropy

                #logging
                avg_loss += loss.mean().item()
                avg_value_loss += value_loss.mean().item()
                avg_entropy += dist_entropy.mean().item()
                count+=1

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.writer.add_scalar('Loss/policy_loss', avg_loss/count, i_episode)
        self.writer.add_scalar('Loss/value_loss', avg_value_loss / count, i_episode)
        self.writer.add_scalar('Loss/entropy', avg_entropy / count, i_episode)
        self.writer.add_scalar('Loss/learning_rate', self.lr_scheduler.get_lr()[0], i_episode)

def play_against_old_checkpoints(checkpoint_folder, model_class, every_n_checkpoint, runs, summary_writer):
    generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
    if len(generations) > 1:
        max_gen = max(generations)
        for i in generations:
            if i != max_gen and i%every_n_checkpoint==0:
                policy_old = model_class()
                policy_old.to(device=device)
                policy_old.load_state_dict(torch.load(checkpoint_folder + "/" + str(i).zfill(8) + ".pt"))

                policy_new = model_class()
                policy_new.to(device=device)
                policy_new.load_state_dict(torch.load(checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))

                gs = Game_Simulation(policy_old, policy_new, policy_old, policy_new, 1)
                all_rewards = np.array([0., 0., 0., 0.])
                for j in range(runs):
                    game_state = gs.run_simulation()
                    rewards = np.array(game_state.get_rewards())
                    all_rewards += rewards

                gs = Game_Simulation(policy_new, policy_old, policy_new, policy_old, 1)
                all_rewards = all_rewards[[1, 0, 3, 2]]
                for j in range(runs):
                    game_state = gs.run_simulation()
                    #gs.print_game(game_state)
                    rewards = np.array(game_state.get_rewards())
                    all_rewards += rewards


                print(str(max_gen) + " vs " + str(i) + " = "+ str(all_rewards[0] + all_rewards[2]) + ":"+ str(all_rewards[1] + all_rewards[3]) +"\n")
                summary_writer.add_scalar('Evaluation/generation_'+str(max_gen), all_rewards[0] + all_rewards[2], i)


def main():

    ############## Hyperparameters ##############
    max_episodes = 9000000  # max training episodes

    update_timestep = 2000  # update policy every n games
    save_checkpoint_every_n = 10000 #save checkpoints every n games
    evaluate_timestep = 50000 #needs to be a multiple of save_checkpoint_every_n
    eval_games = 200
    checkpoint_folder = "policies"

    lr = 0.0001
    lr_stepsize = 200000
    lr_gamma = 0.1

    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 8  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    c1, c2 = 0.5, 0.05
    batch_size = 400
    random_seed = None
    #############################################

    model = ActorCriticNetwork5

    # creating environment
    if random_seed:
        torch.manual_seed(random_seed)

    #loading initial policy
    policy = model().to(device)
    # take the newest generation available
    # file pattern = policy-000001.pt
    max_gen = 0
    generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
    if len(generations) > 0:
        max_gen = max(generations)
        policy.load_state_dict(torch.load(checkpoint_folder+"/" + str(max_gen).zfill(8) + ".pt"))

    #create ppo
    ppo = PPO(policy, [lr, lr_stepsize, lr_gamma], betas, gamma, K_epochs, eps_clip, batch_size, c1=c1, c2=c2)

    #create a game simulation
    gs = Game_Simulation(ppo.policy_old, ppo.policy_old, ppo.policy_old, ppo.policy_old)  #<------------------------------------remove seed

    # training loop
    for i_episode in range(max_gen+1, max_episodes + 1):

        # Running policy_old:
        t0 = time.time_ns()
        game_state = gs.run_simulation()
        t1 = time.time_ns()


        # update if its time
        if i_episode % update_timestep == 0:
            t2 = time.time_ns()
            ppo.update(gs.get_memory(), i_episode)
            t3 = time.time_ns()
            ppo.lr_scheduler.step(i_episode)

            # logging
            print("Episode: "+str(i_episode) + " game simulation (ms) = "+str((t1-t0)/1000000) + " update (ms) = "+str((t3-t2)/1000000))
            gs.print_game(game_state) #<------------------------------------remove
            ppo.writer.add_scalar('Games/weiter', gs.game_count[0]/update_timestep, i_episode)
            ppo.writer.add_scalar('Games/sauspiel', gs.game_count[1] / update_timestep, i_episode)
            ppo.writer.add_scalar('Games/wenz', gs.game_count[2] / update_timestep, i_episode)
            ppo.writer.add_scalar('Games/solo', gs.game_count[3] / update_timestep, i_episode)

            ppo.writer.add_scalar('WonGames/sauspiel', np.divide(gs.won_game_count[1], gs.game_count[1]), i_episode)
            ppo.writer.add_scalar('WonGames/wenz', np.divide(gs.won_game_count[2],gs.game_count[2]), i_episode)
            ppo.writer.add_scalar('WonGames/solo', np.divide(gs.won_game_count[3],gs.game_count[3]), i_episode)

            # reset memories and replace policy
            gs = Game_Simulation(ppo.policy_old, ppo.policy_old, ppo.policy_old, ppo.policy_old)


        # evaluation
        if i_episode % save_checkpoint_every_n == 0:
            print("Saving Checkpoint")
            torch.save(ppo.policy_old.state_dict(), checkpoint_folder + "/" + str(i_episode).zfill(8) + ".pt")
            print("Evaluation")
            if i_episode % evaluate_timestep == 0:
                play_against_old_checkpoints(checkpoint_folder, model,evaluate_timestep,eval_games,ppo.writer)


if __name__ == '__main__':
    main()
