import torch
from torch.utils import data

class ExperienceDatasetLinear(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, logprobs, rewards):
        'Initialization'
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.rewards = rewards



  def __len__(self):
        'Denotes the total number of samples'
        return len(self.actions)

  def __getitem__(self, index):
        'Generates one sample of data'
        return [self.states[index], self.actions[index], self.logprobs[index], self.rewards[index]]

  def custom_collate(self, batch):

    states_batch, actions_batch, logprobs_batch, rewards_batch = zip(*batch)

    #states = [state[0] for state in states_batch]
    #states = torch.stack(states).detach()

    states = []
    transposed_states = list(map(list, zip(*states_batch)))
    states.append(torch.stack(transposed_states[0]).detach())
    states.append(torch.stack(transposed_states[1]).detach())

    actions = torch.stack(actions_batch).detach()
    logprobs = torch.stack(logprobs_batch).detach()
    rewards = torch.tensor(rewards_batch)

    return [states, actions, logprobs, rewards]

class ExperienceDatasetLSTM(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, logprobs, rewards, num_sequences=2):
        'Initialization'
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.rewards = rewards
        self.num_sequences = num_sequences



  def __len__(self):
        'Denotes the total number of samples'
        return len(self.actions)

  def __getitem__(self, index):
        'Generates one sample of data'
        return [self.states[index], self.actions[index], self.logprobs[index], self.rewards[index]]

  def custom_collate(self, batch):

      states_batch, actions_batch, logprobs_batch, rewards_batch = zip(*batch)

      # convert list to tensor
      # torch.stack(memory.states).to(device).detach()
      states = []
      transposed_states = list(map(list, zip(*states_batch)))
      states.append(torch.stack(transposed_states[0]).detach())

      for i in range(self.num_sequences):
          sequences = [torch.squeeze(seq, dim=1).detach() for seq in transposed_states[1+i]]
          seq_lengths = [len(x) for x in sequences]
          # pad the seq_batch
          padded_seq_batch = torch.nn.utils.rnn.pad_sequence(sequences)
          # pack the padded_seq_batch
          packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lengths,
                                                                     enforce_sorted=False)

          states.append(packed_seq_batch)

      states.append(torch.stack(transposed_states[-1]).detach())

      actions = torch.stack(actions_batch).detach()
      logprobs = torch.stack(logprobs_batch).detach()
      rewards = torch.tensor(rewards_batch).float()

      return [states, actions, logprobs, rewards]

class PredictionDatasetLSTM(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, states, predictions, num_sequences=1):
        'Initialization'
        self.states = states
        self.predictions = predictions
        self.num_sequences = num_sequences

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.states)

  def __getitem__(self, index):
        'Generates one sample of data'
        return [self.states[index], self.predictions[index]]

  def custom_collate(self, batch):

      states_batch, predictions_batch = zip(*batch)

      # convert list to tensor
      # torch.stack(memory.states).to(device).detach()
      states = []
      transposed_states = list(map(list, zip(*states_batch)))
      states.append(torch.stack(transposed_states[0]).detach())

      for i in range(self.num_sequences):
          sequences = [torch.squeeze(seq, dim=1).detach() for seq in transposed_states[1+i]]
          seq_lengths = [len(x) for x in sequences]
          # pad the seq_batch
          padded_seq_batch = torch.nn.utils.rnn.pad_sequence(sequences)
          # pack the padded_seq_batch
          packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lengths,
                                                                     enforce_sorted=False)
          states.append(packed_seq_batch)

      states.append(torch.stack(transposed_states[len(transposed_states)-1]).detach())

      predictions = torch.stack(predictions_batch).detach()


      return [states, predictions]

