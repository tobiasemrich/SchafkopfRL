import torch
from torch.utils import data

class ExperienceDatasetLSTM(data.Dataset):

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

    # convert list to tensor
    # torch.stack(memory.states).to(device).detach()
    states = []
    transposed_states = list(map(list, zip(*states_batch)))
    states.append(torch.stack(transposed_states[0]).detach())

    for i in range(2):
        sequences = [torch.squeeze(seq, dim=1).detach() for seq in transposed_states[1+i]]
        seq_lengths = [len(x) for x in sequences]
        # pad the seq_batch
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(sequences)
        # pack the padded_seq_batch
        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lengths,
                                                                   enforce_sorted=False)

        states.append(packed_seq_batch)

    actions = torch.stack(actions_batch).detach()
    allowed_actions = torch.stack(allowed_actions_batch).detach()
    logprobs = torch.stack(logprobs_batch).detach()
    rewards = torch.tensor(rewards_batch)

    return [states, actions, allowed_actions, logprobs, rewards]