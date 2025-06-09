import subprocess

# Define the commands
commands = [
    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "PPO"',
    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "PPO"',

    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 334 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "PPO_env"',

    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 101 --noise_seed 25 --buffer_seed 5 --model_name "PPO_policy"',

    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 26 --buffer_seed 5 --model_name "PPO_noise"',

    # 'python PPOscript.py --learning_timesteps 25000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 6 --model_name "PPO_buffer"',


    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "SAC"',
    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "SAC"',

    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 334 --policy_seed 100 --noise_seed 25 --buffer_seed 5 --model_name "SAC_env"',

    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 101 --noise_seed 25 --buffer_seed 5 --model_name "SAC_policy"',

    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 26 --buffer_seed 5 --model_name "SAC_noise"',

    # 'python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 6 --model_name "SAC_buffer"'

    'python superOracle.py --algorithm SAC --sections_number 3 --learning_timesteps 10000 --env_name Pendulum-v1 --policy_seed 111 --noise_seed 222 --env_seed 333 --parameter buffer --seed_range 777,777',

    'python superOracle.py --algorithm PPO --sections_number 3 --learning_timesteps 25000 --env_name Pendulum-v1 --policy_seed 111 --noise_seed 222 --env_seed 333 --parameter buffer --seed_range 777,777'


]

# Execute each command sequentially
for command in commands:
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the process to complete before moving to the next one
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}: {command}")
    else:
        print(f"Command succeeded: {command}")

print("All commands executed.")
