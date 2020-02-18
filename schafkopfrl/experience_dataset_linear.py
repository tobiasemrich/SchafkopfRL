import torch
from torch.utils import data

class ExperienceDatasetLinear(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, allowed_actions, logprobs, rewards):
        'Initialization'
        self.states = states
        self.actions = actions
        self.allowed_actions = allowed_actions
        self.logprobs = logprobs
        self.rewards = rewards



  def __len__(self):
        'Denotes the total number of samples'
        return len(self.actions)

  def __getitem__(self, index):
        'Generates one sample of data'
        return [self.states[index], self.actions[index], self.allowed_actions[index], self.logprobs[index], self.rewards[index]]

def custom_collate(batch):

    states_batch, actions_batch, allowed_actions_batch, logprobs_batch, rewards_batch = zip(*batch)

    states = [state[0] for state in states_batch]
    states = torch.stack(states).detach()

    actions = torch.stack(actions_batch).detach()
    allowed_actions = torch.stack(allowed_actions_batch).detach()
    logprobs = torch.stack(logprobs_batch).detach()
    rewards = torch.tensor(rewards_batch)

    return [[states], actions, allowed_actions, logprobs, rewards]