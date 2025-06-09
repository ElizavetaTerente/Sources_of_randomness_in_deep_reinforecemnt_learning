import argparse
import os
import subprocess
import sys
from common import (
    find_model_with_name_contains,
    delete_folder,
    ORACLE_SCRIPT,
    MODELS_DIR,
    TEMP_MODELS_DIR,
    BUFFER_DIR,
    NOISE_DIR,
    COMMAND
)

from analyse.graph import draw_single_graph

"""
This script :

Divides learning process into sections (receives number of sections as an input parameter) 
and iteratively runs the oracle script for each section.
For the first section new model in oracle created.
For section > 1 it finds the best(worst)-performing model from the previous oracle 
and passes it as an input parameter for the next run.

Functions:
----------
- run_oracle(anti, algorithm, model_previous_step, section, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed, parameter, seed_range):
    Executes the 'oracle.py' script with the provided parameters using subprocess.run.
    Includes the best/worst model from the previous step for if section > 1.

- main(anti, algorithm, sections_number, learning_timesteps, env_name, env_seed, policy_seed, noise_seed, buffer_seed, parameter, seed_range):
    Divides the total learning timesteps into sections and iteratively runs the oracle script for each section.
    After section > 1 it finds and sets the best/worst model from the previous section to be used in the next section.

Usage:
------
To run the script from the command line, use the following command:
    python superOracle.py --anti ANTI --algorithm {SAC} --sections_number SECTIONS_NUMBER --learning_timesteps LEARNING_TIMESTEPS
    --env_name ENV_NAME --env_seed ENV_SEED --policy_seed POLICY_SEED --noise_seed NOISE_SEED --buffer_seed  
    BUFFER_SEED --parameter {env,policy,noise,buffer} --seed_range SEED_RANGE

Concrete example : 
    python superOracle.py --anti False--algorithm SAC --sections_number 3 --learning_timesteps 30000 
    --env_name "Pendulum-v1" --policy_seed 100 --noise_seed 25 --buffer_seed 1  
    --parameter env --seed_range 6,7


Arguments:
----------
--anti: if True - choose the worst-performed model, 
if False - best-performed
--algorithm: The RL algorithm to test (SAC,PPO).
--sections_number: Total number of sections. Oracle executed for every section.
--learning_timesteps: Total number of timesteps for training.
--env_name: Environment to test in.
--env_seed: Seed for the environment.
--policy_seed: Seed for the policy weights and biases.
--noise_seed: Seed for the action noise generator.
--buffer_seed: Seed for sampling from the replay buffer.
--parameter: Parameter to test (choices: 'env', 'policy', 'noise', 'buffer').
--seed_range: Range of values to test for the seed (format: start,end).
"""

def run_oracle(
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
    
    current_directory = os.getcwd()
    script_path = os.path.join(current_directory, ORACLE_SCRIPT)
    script_dir = os.path.dirname(script_path)

    # Command to run the script
    command = [
        sys.executable,
        script_path,
        "--anti",
        anti,
        "--algorithm",
        algorithm,
        "--section",
        str(section),
        "--learning_timesteps",
        str(learning_timesteps),
        "--env_name",
        str(env_name),
        "--parameter",
        str(parameter),
        "--seed_range",
        str(seed_range)
    ]

    seeds = [
        (env_seed, "--env_seed"),
        (policy_seed, "--policy_seed"),
        (noise_seed, "--noise_seed"),
        (buffer_seed, "--buffer_seed")
    ]

    # Count the number of None arguments
    none_count = sum(1 for arg, _ in seeds if arg is None)
    # Check if more than one argument is None
    if none_count > 1:
        raise ValueError("More than one of the optional arguments (env_name, env_seed, policy_seed, noise_seed, buffer_seed) is None")
    
    # Conditionally add optional arguments
    for arg, name in seeds:
        if arg is not None:
            command.extend([name, str(arg)])

    # Conditionally add model_previous_step if it's not None
    if model_previous_step is not None:
        command.extend(["--model_previous_step", model_previous_step])

    print(f"Running command: {command}")

    try:
        result = subprocess.run(
            command, cwd=script_dir, capture_output=True, text=True, check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running : {e}")
        print(e.stdout)
        print(e.stderr)

def main(
    anti,
    algorithm,
    sections_number,
    learning_timesteps,
    env_name,
    env_seed,
    policy_seed,
    noise_seed,
    buffer_seed,
    parameter,
    seed_range,
):
    if  algorithm.lower() == "ppo" and parameter.lower() == "policy" and sections_number != 1:
        raise ValueError("Policy investigation in PPO is only possible with sections_number = 1")
    
    # Save the command line arguments to a file
    with open(COMMAND, "w") as command_file:
        command_file.write(" ".join(sys.argv) + "\n")

    model_previous_step = None
    learning_timesteps_per_section = round(learning_timesteps / sections_number)
    for i in range(1, sections_number + 1):
        run_oracle(anti,algorithm,model_previous_step,i,learning_timesteps_per_section,env_name,env_seed,policy_seed,noise_seed,buffer_seed,parameter,seed_range)

        nameContains = (
                    f"{algorithm}_{i}_{learning_timesteps_per_section}_{env_name}_"
                    f"{env_seed}_{policy_seed}_{noise_seed}_{buffer_seed}_{parameter}="
                )
        
        # Find the model
        model_previous_step= find_model_with_name_contains(MODELS_DIR, nameContains)
        
        if i == sections_number:
            delete_folder(TEMP_MODELS_DIR)
            delete_folder(BUFFER_DIR)
            delete_folder(NOISE_DIR)
            draw_single_graph(anti=anti.lower() == "true",model_name=model_previous_step, sections_number=sections_number)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL algorithm with varying seeds")
    parser.add_argument(
        "--anti",
        type=str,
        required=True,
        choices=["True", "False"],
        help="specify if to run supeOracle or antiSuperOracle",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["SAC","PPO"],
        help="Algorithm to test (SAC,PPO)",
    )
    parser.add_argument(
        "--sections_number", type=int, required=True, help="Total number of sections. Oracle executed for every section."
    )
    parser.add_argument(
        "--learning_timesteps", type=int, required=True, help="Total number of timesteps for trainig"
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
        help="Range of values to test for the seed (start,end)",
    )

    args = parser.parse_args()

    main(
        anti = args.anti,
        algorithm=args.algorithm,
        sections_number=args.sections_number,
        learning_timesteps=args.learning_timesteps,
        env_name=args.env_name,
        env_seed=args.env_seed,
        policy_seed=args.policy_seed,
        noise_seed=args.noise_seed,
        buffer_seed=args.buffer_seed,
        parameter=args.parameter,
        seed_range=args.seed_range,
    )
