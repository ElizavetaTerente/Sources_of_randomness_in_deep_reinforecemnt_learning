# Sources of randomness in Deep Reinforecemnt Learning

This repository contains the experimental setup used to collect the datasets and the resulting datasets by itself (_result.csv_ folder). A detailed description of the process, results, and conclusions can be found in the accompanying report and in the presentation slides (_documents_ folder).

## Theory

Deep Reinforcement Learning (DRL) is a highly advanced area in artificial intelligence where agents
learn optimal behaviors through trial and error, guided by complex neural network architectures.
The effectiveness of the strategies learned by the agent can be quantified through its ability to learn
and adapt to its environment, which is referred to as performance.

A fundamental aspect of DRL training involves the agent exploring its environment through
strategies introducing randomness into the decision-making process. While beneficial for exploring
a wide range of possible actions and states, this randomness also means that the learning process
can be unpredictable, leading to very different results even if the training conditions are the same.

The randomness is controlled by a random number generator that uses a seed, which is a starting
value that determines the sequence of random numbers generated and thus influences the random
decision-making. Explicitly setting the same seed ensures that the learning process follows an identical path, resulting in the same outcomes. This approach allows us to achieve reproducibility.
Conversely, using different seeds will result in varying outcomes.

The purpose of this research is to explore and define the impact of different sources of
randomness on performance.

## What was done

This research involved designing an experimental setup, conducting experiments, and collecting
a dataset for analysis. As a result, the gathered data enabled conclusions to be drawn about the
impact on the performance of each source of randomness and quantitatively compare them.

The experimentation focused on identifying the best and worst performance scenarios for each
source of randomness independently. To achieve this, the algorithms of use were modified to allow separate seeds for each source. By fixing all sources (their seeds) and varying only the seed of the one we want to investigate, it was possible to isolate and study how that specific
source influenced performance.

After identifying the best and worst performance cases, the corresponding learning curves were
compared to analyze their differences. To quantitatively assess the divergence between them, the
Symmetric Mean Absolute Percentage Error (SMAPE) metric, explained in detail in section 4.1,
was used. A high SMAPE value (in %) indicates a significant difference between the best and worst
performance curves, suggesting that variations in the seed for a particular source of randomness can
greatly influence the learning process and performance, either enhancing or degrading it. In contrast,
a low SMAPE value signifies minimal divergence, meaning that different seeds for the source under
investigation have little impact on the learning process.

The experiments were conducted across various algorithms and environments.

## Conclusion
Based on the collected data, it was determined that 75% of the time, the impact (quantified
using SMAPE) of any source of randomness will not exceed 45%. This implies that by running the
same algorithm under identical training conditions, without explicitly setting the seed, there is a
75% probability that the difference in performance will remain below 45% (SMAPE).

It was also discovered that 65% of the time, the impact (quantified using SMAPE) of the sources
in the current environment is nearly equal (within a 10% SMAPE range).

The highest recorded impact reached 92% (SMAPE), while the lowest impact accounted for only
0.6% (SMAPE).

Overall, the sources can be grouped by their SMAPE-based impact, with the first group showing
a slightly stronger influence than the second. However, a comprehensive analysis of the collected
data and its mean values indicate that the impact of each source is relatively similar, with only
minor differences.

## Getting Started

### superOracle.py

Divides learning process into sections (receives number of sections as an input parameter) and iteratively runs the oracle script for each section.
For the first section new model in oracle created.
For section > 1 it finds the best(worst)-performing model from the previous oracle and passes it as an input parameter for the next run.

### oracle.py

The script to run experiments. This script trains SAC and PPO models in a given environment with a specified seed range for the chosen source of randomness, while keeping fixed seeds for other sources of randomness. The models are then evaluated based on the average reward. Information about the best(worst)-performing model is saved in `results.csv`. Intermediate results from evaluating each model (from which the best(worst) one is later chosen) are saved in `backlog_evaluation.csv`. The script executes trainings in parallel. In the end, all models except the best(worst)-performing one deleted.

The main script to run experiments. Trains SAC or PPO models in parallel in a given environmnet 
with given seed range for the chosen source of randomness 
and fixed seeds for other sources of randomness.
->
Models being evaluated based on the average reward.
Information about evaluation results (model name and 
average reward) saved in `backlog_evaluation.csv`.
All the models and corresponding buffers saved in `/models` folder.
->
Corresponding informaiton about best(worst) performed model 
(algorithm, environment, env_seed, policy_seed, noise_seed, 
buffer_seed, parameter, seed_value, model_name, avg_reward)
saved in `results.csv`.
->
All the models and corresponding files except the best(worst)-performed one cleaned.

### Other Scripts Called from Oracle.py

#### SACscript.py

Trains a Soft Actor-Critic (SAC) model using custom seeds for the environment, policy weights and biases, action noise generator, and replay buffer sampling. Seed values are passed as command-line parameters. The trained model is evaluated and saved. Result written in `backlog_evaluation.csv`.

#### PPOscript.py

Trains a Proximal Policy Optimization (PPO) model using custom seeds for the environment, policy weights and biases, action noise generator, and rollout buffer sampling. Seed values are passed as command-line parameters. The trained model is evaluated and saved. Result written in `backlog_evaluation.csv`.

### General Example of Running superOracle

```
python superOracle.py --anti {False,True} --algorithm {SAC,PPO} --sections_number SECTIONS_NUMBER --learning_timesteps LEARNING_TIMESTEPS --env_name ENV_NAME --env_seed ENV_SEED --policy_seed POLICY_SEED --noise_seed NOISE_SEED --buffer_seed BUFFER_SEED --parameter {env,policy,noise,buffer} --seed_range SEED_RANGE
```

Where:
- `ANTI`: if True - choose the worst-performed model, 
if False - best-performed
- `ALGORITHM`: "SAC" or "PPO"
- `SECTIONS_NUMBER`: Total number of sections. Oracle executed for every section.
- `LEARNING_TIMESTEPS`: Total number of timesteps for training.
- `ENV_NAME`: Name of the environment as a string
- `ENV_SEED`: Seed for environment
- `POLICY_SEED`: Seed for policy weights and biases
- `NOISE_SEED`: Seed for action noise generator
- `BUFFER_SEED`: Seed for sampling from the buffer
- `PARAMETER`: 
  - `env`: environment
  - `policy`: policy weights and biases
  - `noise`: action noise generator
  - `buffer`: sampling from the replay buffer (SAC) or sampling from the rollout buffer (PPO)
- `SEED_RANGE`: Range of seeds to try for the current source of randomness

### General Example of Running Oracle

```
python oracle.py --anti False--algorithm ALGORITHM --env_name ENV_NAME --env_seed ENV_SEED --policy_seed POLICY_SEED --noise_seed NOISE_SEED --buffer_seed BUFFER_SEED --parameter PARAMETER --seed_range SEED_RANGE
```

Where:
- `ANTI`: if True - choose the worst-performed model, 
if False - best-performed
- `ALGORITHM`: "SAC" or "PPO"
- `ENV_NAME`: Name of the environment as a string
- `ENV_SEED`: Seed for environment
- `POLICY_SEED`: Seed for policy weights and biases
- `NOISE_SEED`: Seed for action noise generator
- `BUFFER_SEED`: Seed for sampling from the buffer
- `PARAMETER`: 
  - `env`: environment
  - `policy`: policy weights and biases
  - `noise`: action noise generator
  - `buffer`: sampling from the replay buffer (SAC) or sampling from the rollout buffer (PPO)
- `SEED_RANGE`: Range of seeds to try for the current source of randomness

### Concrete Examples

## superOracle

```
python superOracle.py --anti False--algorithm SAC --sections_number 3 --learning_timesteps 30000 --env_name "Pendulum-v1" --policy_seed 100 --noise_seed 25 --buffer_seed 1 --parameter env --seed_range 6,7
```

```
python superOracle.py --anti True --algorithm PPO --sections_number 3 --learning_timesteps 30000 --env_name "Pendulum-v1" --policy_seed 100 --noise_seed 25 --buffer_seed 1 --parameter env --seed_range 6,7
```

***Result of running superOracle and AntiSuperOracle u can see in "exampleOutput" folder.***

## Oracle

```
python oracle.py --anti True --algorithm SAC --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 1 --parameter env --seed_range 6,7
```

```
python oracle.py --anti Flase --algorithm PPO --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 1 --parameter env --seed_range 6,7
```

## SACscript

```
python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 335 --policy_seed 102 --noise_seed 25 --buffer_seed 1 --model_name "SAC_Pendulum"
```
## PPOscript

```
python SACscript.py --learning_timesteps 10000 --env_name "Pendulum-v1" --env_seed 333 --policy_seed 100 --noise_seed 25 --buffer_seed 1 --model_name "SAC_Pendulum"
```

