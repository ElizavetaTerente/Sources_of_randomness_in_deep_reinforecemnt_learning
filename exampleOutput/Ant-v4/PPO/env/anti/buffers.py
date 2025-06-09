from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize
from typing import Generator


#SAC
class CustomReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        seed = None
    ):
        super(CustomReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.device = device
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = self.rng.integers(0, upper_bound, size=batch_size)
        
        batch = self._get_samples(batch_inds, env=env)
        return ReplayBufferSamples(
            observations=batch.observations.to(self.device),
            actions=batch.actions.to(self.device),
            next_observations=batch.next_observations.to(self.device),
            dones=batch.dones.to(self.device),
            rewards=batch.rewards.to(self.device),
        )
    
    def set_sampling_seed(self, seed):
        self.rng = np.random.default_rng(seed)

#PPO
class CustomRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1,
        seed=None,
    ):
        super(CustomRolloutBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gamma,
            gae_lambda,
            n_envs,
        )
        self.device = device
        self.rng = np.random.default_rng(seed)
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = self.rng.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def set_sampling_seed(self, seed):
        self.rng = np.random.default_rng(seed)