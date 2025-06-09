import torch
import numpy as np
from pink import PinkNoiseDist
from pink import ColoredNoiseProcess
from cnppo.sb3 import ColoredNoiseDist

import torch
import numpy as np

#SAC
class CustomPinkNoiseDist(PinkNoiseDist):
    def __init__(self, seq_len, action_dim, rng=None, epsilon=1e-6):
        super().__init__(seq_len, action_dim, rng, epsilon)
        self.device = torch.device("cuda")
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.rng = rng if rng is not None else np.random.default_rng()

    def to(self, device):
        self.device = device
        return self

    def sample(self) -> torch.Tensor:
        if np.isscalar(self.beta):
            cn_sample = torch.tensor(self.gen.sample()).float().to(self.device)
        else:
            cn_sample = (
                torch.tensor([cnp.sample() for cnp in self.gen]).float().to(self.device)
            )
        self.gaussian_actions = (
            self.distribution.mean.to(self.device)
            + self.distribution.stddev.to(self.device) * cn_sample
        )
        return torch.tanh(self.gaussian_actions)

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._initialize_noise_generator()

    def _initialize_noise_generator(self):
        if np.isscalar(self.beta):
            self.gen = ColoredNoiseProcess(beta=self.beta, size=(self.action_dim, self.seq_len), rng=self.rng)
        else:
            self.gen = [ColoredNoiseProcess(beta=b, size=self.seq_len, rng=self.rng) for b in self.beta]

    def print_sample(self):
        sample = self.sample()
        print("Generated Sample:", sample)

#PPO
class CustomColoredNoiseDist(ColoredNoiseDist):
    def __init__(self, beta, seq_len, action_dim=None, rng=None, action_low=None, action_high=None):
        super().__init__(beta, seq_len, action_dim, rng, action_low, action_high)
        self._initial_rng = rng if rng is not None else np.random.default_rng()
        self.rng = self._initial_rng
        self._gens = {}

    def to(self, device):
        self.device = device
        return self

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._gens = {}
    
    def get_gen(self, n_envs, device="cpu"):
        if n_envs not in self._gens:
            if np.isscalar(self.beta):
                gen = ColoredNoiseProcess(beta=self.beta, size=(n_envs, self.action_dim, self.seq_len), rng=self.rng)
            else:
                gen = [ColoredNoiseProcess(beta=b, size=(n_envs, self.seq_len), rng=self.rng) for b in self.beta]
            self._gens[n_envs] = gen
        return self._gens[n_envs]
    
    def print_sample(self):
        sample = self.sample()
        print("Generated Sample:", sample)


