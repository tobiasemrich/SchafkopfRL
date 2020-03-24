import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import  StepLR

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from schafkopfrl import experience_dataset_linear, experience_dataset_lstm
from schafkopfrl.experience_dataset_lstm import ExperienceDatasetLSTM
from schafkopfrl.experience_dataset_linear import ExperienceDatasetLinear

import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, policy, lr_params, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch_size, c1=0.5, c2=0.01, start_episode = -1):
        [self.lr, self.lr_stepsize, self.lr_gamma] = lr_params

        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.batch_size= batch_size
        self.mini_batch_size = mini_batch_size

        if batch_size % mini_batch_size != 0:
            raise Exception("batch_size needs to be a multiple of mini_batch_size")

        self.c1 = c1
        self.c2 = c2

        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.lr, betas=betas, weight_decay=5e-5)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.lr_stepsize, gamma=self.lr_gamma)
        self.lr_scheduler.step(start_episode)
        self.policy_old = type(policy)().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.writer = SummaryWriter(log_dir="../runs")

        self.logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)

    def update(self, memory, i_episode):

        self.policy.train()

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #rewards.insert(0, discounted_reward)
            rewards.append(discounted_reward)
        rewards.reverse()


        self.logger.info("AVG rewards: "+ str(np.mean(rewards)))
        self.logger.info("STD rewards: " + str(np.std(rewards)))
        # Normalizing the rewards:
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)



        # Create dataset from collected experiences
        #experience_dataset = ExperienceDatasetLinear(memory.states, memory.actions, memory.allowed_actions, memory.logprobs, rewards)
        experience_dataset = ExperienceDatasetLSTM(memory.states, memory.actions, memory.allowed_actions,
                                                     memory.logprobs, rewards)

        #training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset_linear.custom_collate, batch_size=self.batch_size, shuffle=True)
        training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset_lstm.custom_collate, batch_size=self.mini_batch_size, shuffle=True)


        # Optimize policy for K epochs:
        avg_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        avg_clip_fraction = 0
        avg_approx_kl_divergence = 0
        avg_explained_var = 0
        count = 0
        for epoch in range(self.K_epochs): #epoch

            mini_batches_in_batch = int(self.batch_size / self.mini_batch_size)
            self.optimizer.zero_grad()

            for i, (old_states, old_actions, old_allowed_actions, old_logprobs, old_rewards) in enumerate(training_generator): # mini batch

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
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                value_loss = self.MseLoss(state_values, old_rewards)
                loss = -torch.min(surr1, surr2) + self.c1 * value_loss - self.c2 * dist_entropy

                clip_fraction = (abs(ratios - 1.0) > self.eps_clip).type(torch.FloatTensor).mean()
                approx_kl_divergence = .5 * ((logprobs - old_logprobs.detach()) ** 2).mean()
                explained_var = 1-torch.var(old_rewards - state_values) / torch.var(old_rewards)

                #logging losses only in the first epoch, otherwise they will be dependent on the learning rate
                #if epoch == 0:
                avg_loss += loss.mean().item()
                avg_value_loss += value_loss.mean().item()
                avg_entropy += dist_entropy.mean().item()
                avg_clip_fraction += clip_fraction.item()
                avg_approx_kl_divergence += approx_kl_divergence.item()
                avg_explained_var += explained_var.mean().item()
                count+=1

                loss.mean().backward()

                if (i + 1) % mini_batches_in_batch == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.writer.add_scalar('Loss/policy_loss', avg_loss/count, i_episode)
        self.writer.add_scalar('Loss/value_loss', avg_value_loss / count, i_episode)
        self.writer.add_scalar('Loss/entropy', avg_entropy / count, i_episode)
        self.writer.add_scalar('Loss/learning_rate', self.lr_scheduler.get_lr()[0], i_episode)
        self.writer.add_scalar('Loss/ppo_clipping_fraction', avg_clip_fraction/count, i_episode)
        self.writer.add_scalar('Loss/approx_kl_divergence', avg_approx_kl_divergence / count, i_episode)
        self.writer.add_scalar('Loss/avg_explained_var', avg_explained_var / count, i_episode)
