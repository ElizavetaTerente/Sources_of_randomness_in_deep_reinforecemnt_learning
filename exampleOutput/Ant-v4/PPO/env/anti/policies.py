from stable_baselines3.sac.policies import MlpPolicy
import torch
from noise import CustomColoredNoiseDist
from cnppo.cnpolicy import ColoredNoiseActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
import torch
from gymnasium import spaces
from torch import nn
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import numpy as np

class CustomMlpPolicy(MlpPolicy):
    """
    CustomMlpPolicy(MlpPolicy):
        Custom MLP policy with an optional policy seed 
        for initializing weights.
    """
    def __init__(self, *args, policy_seed=None, **kwargs):
            self.policy_seed = policy_seed
            super(CustomMlpPolicy, self).__init__(*args, **kwargs)
            self._initialize_parameters() 

    def _build(self, *args, **kwargs):
        super(CustomMlpPolicy, self)._build(*args, **kwargs)
        if self.policy_seed is not None:
            self._initialize_parameters()

    def _initialize_parameters(self):
        if self.policy_seed is not None:
            torch.manual_seed(self.policy_seed)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def reset(self, seed):
        self.policy_seed = seed
        self._initialize_parameters()

class CustomColoredNoiseActorCriticPolicy(ColoredNoiseActorCriticPolicy):
    """
    CustomColoredNoiseActorCriticPolicy class inherited from ColoredNoiseActorCriticPolicy 
    with an additional parameter to set the seed for policy (weights and biases).
    Uses CustomColoredNoiseDist.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        noise_color_beta: float = 0.0,
        noise_seq_len: int = 1024,
        noise_rng: np.random.Generator = None,
        policy_seed: Optional[int] = None,  # Add policy seed parameter
    ):
        if policy_seed is not None:
            torch.manual_seed(policy_seed)
            np.random.seed(policy_seed)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            noise_color_beta=noise_color_beta,
            noise_seq_len=noise_seq_len,
            noise_rng=noise_rng,
        )

        self.action_dist = CustomColoredNoiseDist(
            beta=self.noise_color_beta,
            seq_len=noise_seq_len,
            action_dim=action_space.shape[0],
            rng=noise_rng,
        )