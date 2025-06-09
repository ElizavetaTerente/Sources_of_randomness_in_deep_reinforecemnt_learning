from stable_baselines3.common.callbacks import BaseCallback
from common import(write_to_csv,
                   FIELDNAMES_REWARDS)

class RewardLoggerCallback(BaseCallback):
    def __init__(self, reward_log_path: str, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.reward_log_path = reward_log_path
        self.last_logged_reward = None

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            last_info = self.model.ep_info_buffer[-1]
            reward = last_info['r']
            if self.last_logged_reward!=reward:
                timestep = self.num_timesteps
                write_to_csv([reward, timestep], self.reward_log_path, FIELDNAMES_REWARDS)
                self.last_logged_reward = reward
        
        return True