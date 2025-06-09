import argparse
import os
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from common import (
    write_to_csv,
    ensure_directory_exists,
    manage_model_log,
    FIELDNAMES_BACKLOG,
    FILENAME_BACKLOG,
    TEMP_MODELS_DIR,
    MODELS_DIR,
    BUFFER_DIR,
    NOISE_DIR,
    LOG_DIR,
    NOISE,
    BUFFER,
    PKL,
    ZIP
)
from common import calculate_average_r
from buffers import CustomRolloutBuffer
from policies import CustomColoredNoiseActorCriticPolicy
import pickle
from callback import RewardLoggerCallback
"""
This script trains a Proximal Policy Optimization (PPO) model using custom 
seeds for environment, policy weights and biases, action noise generator, 
and rollout buffer sampling.

Functions:
----------
main(input_model, learning_timesteps,env_name, env_seed, policy_seed, noise_seed, buffer_seed, buffer_size, model_name):
    Main function to set up the environment, 
    create the PPO model with custom components, 
    train the model, and save the results.

Usage:
------
To run the script from the command line, use the following command:
    python PPOscript.py --learning_timesteps LEARNING_TIMESTEPS 
    --env_name ENV_NAME --env_seed ENV_SEED 
    --policy_seed POLICY_SEED --noise_seed NOISE_SEED 
    --buffer_seed BUFFER_SEED --model_name MODEL_NAME

Concrete example:
    python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 
    --policy_seed 100 --noise_seed 25 --buffer_seed 1 
    --model_name "PPO_Pendulum"

Arguments:
----------
--input_model : optional argument for input model path (Needed for the 2nd and subsequent runs of Oracle from SuperOracle)
--learning_timesteps: Total number of learning timesteps for training
--env_name: The environment to test in.
--env_seed: Seed for the environment.
--policy_seed: Seed for the policy weights and biases.
--noise_seed: Seed for the action noise generator.
--buffer_seed: Seed for sampling from the rollout buffer.
--buffer_size : Size of the buffer with default value of 2048
--model_name: Name to save the trained model.
"""

def main(input_model, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed, buffer_size, model_name):

    vec_env = make_vec_env(
        env_name, n_envs=4, seed=env_seed, wrapper_class=gym.wrappers.FlattenObservation
    )

    # Load PPO (2nd and subsequent Oracle run)
    if input_model:
        if not os.path.isfile(os.path.join(MODELS_DIR, input_model)):
            raise FileNotFoundError(f"Model file '{input_model}' does not exist")
        
        #---------------load--------------
        #model
        model = PPO.load(os.path.join(MODELS_DIR, input_model), vec_env)
        print(f"Loaded model from {input_model}")

        #noise distribution
        with open(os.path.join(NOISE_DIR, input_model.replace(ZIP, "")) + NOISE + PKL, "rb") as f:
            model.policy.action_dist = pickle.load(f)

        #buffer
        with open(os.path.join(BUFFER_DIR, input_model.replace(ZIP, "")) + BUFFER + PKL, "rb") as f:
            model.rollout_buffer = pickle.load(f)

        #---------------reset--------------
        #env
        vec_env.seed(seed=env_seed)
        vec_env.reset()

        #noise distribution
        model.policy.action_dist.reset_seed(noise_seed)

        #buffer
        model.rollout_buffer.set_sampling_seed(buffer_seed)

    #create PPO (if run Oracle for the 1st time)
    else:
        model = PPO(
            CustomColoredNoiseActorCriticPolicy,
            vec_env,
            verbose=1,
            policy_kwargs={
                "noise_color_beta": 0.5,
                "noise_rng": np.random.default_rng(noise_seed),
                "policy_seed": policy_seed
            },
        )

        model.policy.action_dist = model.policy.action_dist.to(model.device)

        custom_rollout_buffer = CustomRolloutBuffer(
            buffer_size=buffer_size,
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            n_envs=vec_env.num_envs,
            device=model.device,
            seed=buffer_seed,
        )

        model.rollout_buffer = custom_rollout_buffer
    
    callback = RewardLoggerCallback(reward_log_path=manage_model_log(LOG_DIR,model_name,input_model))

    print("-----------------------start-----------------------")

    # Learn from the environment
    model.learn(total_timesteps=learning_timesteps,reset_num_timesteps=False, callback=callback)
    
    print("------------------------end------------------------")

    #---------------save--------------
    #model
    ensure_directory_exists(TEMP_MODELS_DIR)
    model_path = os.path.join(TEMP_MODELS_DIR, model_name)
    model.save(model_path)

    #noise distribudtion
    noise_path = os.path.join(NOISE_DIR, model_name) + NOISE + PKL
    ensure_directory_exists(NOISE_DIR)
    with open(noise_path, "wb") as f:
        pickle.dump(model.policy.action_dist, f)

    #buffer
    buffer_path = os.path.join(BUFFER_DIR, model_name) + BUFFER + PKL
    ensure_directory_exists(BUFFER_DIR)
    with open(buffer_path, "wb") as f:
        pickle.dump(model.rollout_buffer, f)

    write_to_csv([model_name, calculate_average_r(LOG_DIR, model_name)], FILENAME_BACKLOG, FIELDNAMES_BACKLOG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO training script with custom seeds"
    )
    parser.add_argument(
        "--input_model", type=str, required=False, default=None, help="Name of the input model"
    )
    parser.add_argument(
        "--learning_timesteps", type=int, required=True, help="Total number of learning timesteps for training"
    )
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument(
        "--env_seed", type=int, required=True, help="Seed for the environment"
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        required=True,
        help="Seed for the policy weights and biases",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        required=True,
        help="Seed for the action noise generator",
    )
    parser.add_argument(
        "--buffer_seed", type=int, required=True, help="Seed for the rollout buffer"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name to save the trained model"
    )

    args = parser.parse_args()

    main(
        input_model=args.input_model,
        learning_timesteps = args.learning_timesteps,
        env_name=args.env_name,
        env_seed=args.env_seed,
        policy_seed=args.policy_seed,
        noise_seed=args.noise_seed,
        buffer_seed=args.buffer_seed,
        buffer_size=args.buffer_size,
        model_name=args.model_name,
    )
