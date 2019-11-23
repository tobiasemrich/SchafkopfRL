import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.allowed_actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.allowed_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append_memory(self, memory):
        self.actions.extend(memory.actions)
        self.allowed_actions.extend(memory.allowed_actions)
        self.states.extend(memory.states)
        self.logprobs.extend(memory.logprobs)
        self.rewards.extend(memory.rewards)
        self.is_terminals.extend(memory.is_terminals)

    def __str__(self):
        out = ""
        for s in self.states:
            out += np.array2string((s.cpu().numpy()), separator=', ', max_line_width=10000) + "\n"
        return out