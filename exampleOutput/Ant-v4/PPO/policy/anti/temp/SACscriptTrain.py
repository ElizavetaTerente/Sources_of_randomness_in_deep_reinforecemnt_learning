import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from noise import CustomPinkNoiseDist
from buffers import CustomReplayBuffer
from common import (
    write_to_csv,
    calculate_average_r_old,
    FIELDNAMES_BACKLOG,
    FILENAME_BACKLOG,
    NOISE,
    BUFFER,
    ENV,
    PKL
)
from policies import CustomMlpPolicy
import pickle

"""

This script trains Soft Actor-Critic (SAC) model using custom seeds 
for environment, policy weights and biases, action noise generator, 
and replay buffer sampling.

It includes custom implementations for policy and replay buffer 
to accommodate these seeds.

Functions:
----------
main(input_model, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed, buffer_size, model_name):
    Main function to set up the environment, 
    create the SAC model with custom components, 
    train the model, and save the results.

Usage:
------
To run the script from the command line, use the following command:
    python SACscript.py --learning_timesteps LEARNING_TIMESTEPS
    --env_name ENV_NAME --env_seed ENV_SEED
    --policy_seed POLICY_SEED --noise_seed NOISE_SEED 
    --buffer_seed BUFFER_SEED --model_name MODEL_NAME

Concrete example:
    python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" 
    --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 1 
    --model_name "SAC_Pendulum"

Arguments:
----------
--input_model : optional argument for input model path (Needed for the 2nd and subsequent runs of Oracle from SuperOracle)
--learning_timesteps: Total number of learning timesteps for training
--env_name: The environment to test in.
--env_seed: Seed for the environment.
--policy_seed: Seed for the policy weights and biases.
--noise_seed: Seed for the action noise generator.
--buffer_seed: Seed for sampling from the replay buffer.
--buffer_size : Size of the buffer with default value of 1000000
--model_name: Name to save the trained model.
"""

def main(input_model, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed,buffer_size, model_name):

    env = gym.make(env_name)
    env.reset(seed=env_seed)
    env.action_space.seed(env_seed)
    env.observation_space.seed(env_seed)

    #for action noise
    seq_len = env._max_episode_steps
    action_dim = env.action_space.shape[-1]
    
    model = SAC(
        CustomMlpPolicy, env, verbose=1, policy_kwargs={"policy_seed": policy_seed}
    )
    print("Created a new model")

    # Set action noise
    model.actor.action_dist = CustomPinkNoiseDist(
        seq_len, action_dim, rng=np.random.default_rng(noise_seed)
    ).to(model.device)
    print("model device : " + str(model.device))

    # Create custom replay buffer and set the sampling seed
    custim_replay_buffer = CustomReplayBuffer(
    buffer_size=buffer_size,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=model.device,
    seed=buffer_seed)
        
    # Replace the model's replay buffer with the custom replay buffer
    model.replay_buffer = custim_replay_buffer

    print("-----------------------start-----------------------")

    # Train the model
    model.learn(total_timesteps=learning_timesteps, log_interval=5,reset_num_timesteps=False)

    print("------------------------end------------------------")

    #----------------save-------------

    # save model
    model_path = os.path.join(model_name)
    model.save(model_path)

    #save noise distribudtion
    noise_path = os.path.join(model_name) + NOISE + PKL
    with open(noise_path, "wb") as f:
        pickle.dump(model.actor.action_dist, f)

    # save buffer
    buffer_path = os.path.join(model_name) + BUFFER + PKL
    with open(buffer_path, "wb") as f:
        pickle.dump(model.replay_buffer, f)
    
    #save environment state
    # env_path = os.path.join(model_name) + ENV + PKL
    # with open(env_path, 'wb') as file:
    #     pickle.dump(env, file)
    #----------------load-------------
    # with open(model_name + ENV + PKL, "rb") as file:
    #     env = pickle.load(file)

    env = gym.make(env_name)

    model = SAC.load(model_name, env)

    with open(model_name + NOISE + PKL, "rb") as f:
        model.actor.action_dist = pickle.load(f)

    with open(model_name + BUFFER + PKL, "rb") as f:
        model.replay_buffer = pickle.load(f)

    #----------------reset-------------

    #noise seed reset
    model.actor.action_dist.reset_seed(noise_seed)

    # policy reset + new seed
    model.policy.reset(policy_seed)

    #env seed
    env.reset(seed=env_seed)
    env.action_space.seed(env_seed)
    env.observation_space.seed(env_seed)

    # #buffer seed reset
    model.replay_buffer.set_sampling_seed(buffer_seed)

    print("-----------------------start-----------------------")

    # Train the model
    model.learn(total_timesteps=learning_timesteps, log_interval=5,reset_num_timesteps=False)

    print("------------------------end------------------------")

    write_to_csv([model_name, calculate_average_r_old(model.ep_info_buffer)], FILENAME_BACKLOG, FIELDNAMES_BACKLOG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC training script with custom seeds"
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
        "--buffer_seed",
        type=int,
        required=True,
        help="Seed for sampling from the replay buffer",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000000,
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
