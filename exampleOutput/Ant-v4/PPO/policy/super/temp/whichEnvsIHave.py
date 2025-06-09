import gymnasium as gym
import dm2gym

# Manually register dm2gym environments if needed
from gymnasium.envs.registration import register

# Example for CheetahRun environment
register(
    id='dm2gym:CheetahRun-v0',
    entry_point='dm2gym.envs:CheetahRun',
)

# Print all available environments
envs = gym.envs.registry.keys()
print("Available environments:")
for env in sorted(envs):
    print(env)

