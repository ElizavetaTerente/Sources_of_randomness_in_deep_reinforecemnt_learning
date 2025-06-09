import argparse
import os
import subprocess
import sys
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from common import (
    write_to_csv,
    clean_up,
    FIELDNAMES_RESULTS,
    FIELDNAMES_BACKLOG,
    FILENAME_RESULTS,
    FILENAME_BACKLOG,
    SAC_SCRIPT,
    PPO_SCRIPT

)
import pandas as pd

"""
This script :

1) trains SAC or PPO models in parallel in a given environmnet 
with given seed range for the chosen source of randomness 
and fixed seeds for other sources of randomness.
->
2)Models being evaluated based on the average reward.
Information about evaluation results (model name and 
average reward) saved in backlog_evaluation.csv.
All the models saved in /models folder.
->
3) Corresponding informaiton about best/worst performed model 
(algorithm, section, learning_timesteps, environment, env_seed, policy_seed, noise_seed, 
buffer_seed, parameter, seed_value, model_name, avg_reward, learning_timesteps)
saved in "results.csv".
->
4)Best/words model moved to /models and /tempModels folder cleaned

Functions:
----------

run_algorithm(script_name, env_name, env_seed, policy_seed, 
noise_seed, buffer_seed, parameter, seed_value, model_name):
    Trains the specified RL algorithm with the given parameters 
    and seeds. Runs the training script as a subprocess 
    and captures the output.

main(algorithm, model_previous_step, section, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed,
parameter, seed_range):
    Main function that orchestrates the training, evaluation, 
    and selection of the best/worst model. Trains models in parallel, 
    evaluates them, identifies the best/worst model, 
    and performs cleanup operations.


Usage:
------
To run the script from the command line, use the following command:
    python oracle.py --algorithm ALGORITHM --learning_timesteps LEARNING_TIMESTEPS
    --env_name ENV_NAME --env_seed ENV_SEED --policy_seed POLICY_SEED --noise_seed NOISE_SEED
    --buffer_seed BUFFER_SEED --parameter PARAMETER --seed_range SEED_RANGE

Concrete example : 
    python oracle.py --algorithm SAC --learning_timesteps 10000 
    --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 
    --buffer_seed 1 --parameter env --seed_range 6,7


Arguments:
----------
--algorithm: The RL algorithm to test (SAC or PPO).
--model_previous_step: spesified by superoracle if necessary (for 2. and futher sections)
--section: number of current section (specified when script is called from superOracle)
--learning_timesteps: total number of steps for training 
--env_name: The environment to test in.
--env_seed: Seed for environment
--policy_seed: Seed for policy weights and biases
--noise_seed: Seed for action noise generator
--buffer_seed: Seed for sampling from the buffer
--parameter: The parameter to test/investigate (env, policy, noise, buffer).
    env = environment
    policy = policy weights and biases
    noise = action noise generator
    buffer = sampling from the replay buffer (SAC) 
    or sampling from the rollout buffer (PPO)
--seed_range: Range of values to test for the seed in the format start,end.
"""

# trainig models
def run_algorithm(
    script_name,
    input_model,
    learning_timesteps,
    env_name,
    env_seed,
    policy_seed,
    noise_seed,
    buffer_seed,
    parameter,
    seed_value,
    model_name,
):
    # Set the seed value for the chosen parameter
    if parameter == "env":
        env_seed = seed_value
    elif parameter == "policy":
        policy_seed = seed_value
    elif parameter == "noise":
        noise_seed = seed_value
    elif parameter == "buffer":
        buffer_seed = seed_value

    # Check if any seed is None and throw an exception if so
    if None in (env_seed, policy_seed, noise_seed, buffer_seed):
        raise ValueError("At least one of the parameters (env_seed, policy_seed, noise_seed, buffer_seed) is None.")

    current_directory = os.getcwd()
    script_path = os.path.join(current_directory, script_name)
    script_dir = os.path.dirname(script_path)

    # Command to run the script
    command = [
        sys.executable,
        script_path,
        "--learning_timesteps",
        str(learning_timesteps),
        "--env_name",
        str(env_name),
        "--env_seed",
        str(env_seed),
        "--policy_seed",
        str(policy_seed),
        "--noise_seed",
        str(noise_seed),
        "--buffer_seed",
        str(buffer_seed),
        "--model_name",
        str(model_name),
    ]

    # Conditionally add model_previous_step if it's not None
    if input_model is not None:
        command.extend(["--input_model", input_model])

    print(f"Running command: {command}")

    try:
        result = subprocess.run(
            command, cwd=script_dir, capture_output=True, text=True, check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {model_name}: {e}")
        print(e.stdout)
        print(e.stderr)

def main(
    anti,
    algorithm,
    model_previous_step,
    section,
    learning_timesteps,
    env_name,
    env_seed,
    policy_seed,
    noise_seed,
    buffer_seed,
    parameter,
    seed_range,
):

    # Determine the script to run based on the algorithm
    script_name = SAC_SCRIPT if algorithm.lower() == "sac" else PPO_SCRIPT

    # Prepare the arguments for the runs
    tasks = []
    for seed_value in seed_range:
        model_name = (
            f"{algorithm}_{section}_{learning_timesteps}_{env_name}_"
            f"{env_seed}_{policy_seed}_{noise_seed}_{buffer_seed}_{parameter}={seed_value}"
        )
        tasks.append(
            (   script_name,
                model_previous_step,
                learning_timesteps,
                env_name,
                env_seed,
                policy_seed,
                noise_seed,
                buffer_seed,
                parameter,
                seed_value,
                model_name,
            )
        )

    # Run the scripts in parallel
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(run_algorithm, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()

    evaluation_tasks = [(task[-1]) for task in tasks]  # Extracting model names

    # Determine the best/worst model
    # Read the CSV file into a DataFrame
    df = pd.read_csv(FILENAME_BACKLOG)

    # Filter the DataFrame to include only rows with model names in evaluation_tasks
    filtered_df = df[df[FIELDNAMES_BACKLOG[0]].isin(evaluation_tasks)]

    # Find the row with the highest-lowest average_reward
    if anti.lower() == "true":
        reward_row = filtered_df.loc[filtered_df[FIELDNAMES_BACKLOG[1]].idxmin()]
    else:
        reward_row = filtered_df.loc[filtered_df[FIELDNAMES_BACKLOG[1]].idxmax()]

    # Extract the model name and the highest average reward
    model_name = reward_row[FIELDNAMES_BACKLOG[0]]
    average_reward = reward_row[FIELDNAMES_BACKLOG[1]]

    # Prepare the best model information
    model_info = {
        FIELDNAMES_RESULTS[0] : anti,
        FIELDNAMES_RESULTS[1]: algorithm,
        FIELDNAMES_RESULTS[2] : section,
        FIELDNAMES_RESULTS[3] : learning_timesteps,
        FIELDNAMES_RESULTS[4]: env_name,
        FIELDNAMES_RESULTS[5]: env_seed,
        FIELDNAMES_RESULTS[6]: policy_seed,
        FIELDNAMES_RESULTS[7]: noise_seed,
        FIELDNAMES_RESULTS[8]: buffer_seed,
        FIELDNAMES_RESULTS[9]: parameter,
        FIELDNAMES_RESULTS[10]: model_name.split("=")[-1],
        FIELDNAMES_RESULTS[11]: model_name,
        FIELDNAMES_RESULTS[12]: average_reward,
    }

    write_to_csv(model_info, FILENAME_RESULTS, FIELDNAMES_RESULTS)

    clean_up(model_name)

    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL algorithm with varying seeds")
    parser.add_argument(
        "--anti",
        type=str,
        required=True,
        choices=["True", "False"],
        help="specify if to choose best models and performance or worst",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["SAC", "PPO"],
        help="Algorithm to test (SAC or PPO)",
    )
    parser.add_argument(
        "--model_previous_step", type=str, required=False, default=None, help="spesified by superoracle if necessary (for 2. and futher sections)"
    )    
    parser.add_argument(
        "--section", type=int, required=False, default=0 , help="number of current section"
    )
    parser.add_argument(
        "--learning_timesteps", type=int, required=True, help="Total leraning timesteps for training"
    )
    parser.add_argument(
        "--env_name", type=str, required=True, help="Environment to test in"
    )

    parser.add_argument(
        "--env_seed", type=int, required=False, help="Seed for the environment"
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        required=False,
        help="Seed for the policy weights and biases",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        required=False,
        help="Seed for the action noise generator",
    )
    parser.add_argument(
        "--buffer_seed",
        type=int,
        required=False,
        help="Seed for sampling from the replay buffer",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        choices=["env", "policy", "noise", "buffer"],
        help="Parameter to test",
    )
    parser.add_argument(
        "--seed_range",
        type=str,
        required=True,
        help="Range of values to test for the seed if format start,end",
    )

    args = parser.parse_args()
    start, end = map(int, args.seed_range.split(","))
    seed_range = range(start, end + 1)

    main(
        anti = args.anti,
        algorithm=args.algorithm,
        model_previous_step=args.model_previous_step,
        section=args.section,
        learning_timesteps=args.learning_timesteps,
        env_name=args.env_name,
        env_seed=args.env_seed,
        policy_seed=args.policy_seed,
        noise_seed=args.noise_seed,
        buffer_seed=args.buffer_seed,
        parameter=args.parameter,
        seed_range=seed_range,
    )
