import os
import numpy
import matplotlib.pyplot as plt
import seaborn
import pandas
import glob
import re
import csv
from scipy.interpolate import interp1d

from common import (
    GRAPH_DIR,
    LOG_DIR,
    ZIP,
    FILENAME_RESULTS
)

ANTI = "anti"
SUPER = "super"
COMMAND = "command"

def divide_sections_and_define_seeds(ax, timesteps,sections_number, chosen_seeds_best=None, chosen_seeds_worst=None):
    vline_color = 'black'
    smoothed_color_best = 'mediumvioletred'
    smoothed_color_worst = 'navy'
    max_timestep = max(timesteps)
    section_interval = round(max_timestep / sections_number)

    for i in range(1, sections_number+1):
        if(i != sections_number):
            ax.axvline(x=i * section_interval, color=vline_color, linestyle='--', alpha=0.5)
        
        # Calculate the midpoint for section label placement
        section_start = (i - 1) * section_interval
        section_midpoint = section_start + section_interval / 2
        
        # Place the section label in the center of the section, on the grid below the curve
        ax.text(section_midpoint, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
                f'Section {i}', color='black', ha='center', va='top')
        
        if(chosen_seeds_best is not None):
            if i < len(chosen_seeds_best)+1:
                ax.text(section_midpoint, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, 
                        chosen_seeds_best[i-1], color=smoothed_color_best, ha='left', va='top', fontweight='bold')
        if(chosen_seeds_worst is not None):
            if i < len(chosen_seeds_worst)+1:
                ax.text(section_midpoint, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, 
                        chosen_seeds_worst[i-1], color=smoothed_color_worst, ha='right', va='top', fontweight='bold')

def draw_combined_graphs(path_to_super=SUPER, path_to_anti=ANTI):
    seaborn.set(style="darkgrid")
    plt.figure(figsize=(13, 7))
    ax = plt.gca()  # Get current axes

    # Find the best model's CSV file
    best_model_path = get_csv_path(path_to_super)
    # Find the worst model's CSV file
    worst_model_path = get_csv_path(path_to_anti)

    # Load data and plot for best and worst models
    best_timesteps, best_rewards = load_data_from_csv(path=best_model_path)
    worst_timesteps, worst_rewards = load_data_from_csv(path=worst_model_path)

    if best_timesteps is None or worst_timesteps is None:
        return
    
    algorithm, env_name, seeds, sections_number, parameter, seed_range = extract_command_params(COMMAND)

    chosen_seeds_best = extract_chosen_seeds_from_csv(os.path.join(SUPER,FILENAME_RESULTS), parameter=parameter)
    chosen_seeds_worst = extract_chosen_seeds_from_csv(os.path.join(ANTI,FILENAME_RESULTS), parameter=parameter)

    worst_smoothed_rewards = plot_graph(ax, worst_timesteps, worst_rewards, anti=True)
    best_smoothed_rewards = plot_graph(ax, best_timesteps, best_rewards, anti=False)

    # Ensure timesteps are the same by interpolating
    min_timestep = max(min(best_timesteps), min(worst_timesteps))
    max_timestep = min(max(best_timesteps), max(worst_timesteps))
    common_timesteps = numpy.linspace(min_timestep, max_timestep, num=min(len(best_timesteps), len(worst_timesteps)))

    # Interpolating the rewards to the common timesteps
    best_interp_func = interp1d(best_timesteps, best_smoothed_rewards, kind='linear')
    worst_interp_func = interp1d(worst_timesteps, worst_smoothed_rewards, kind='linear')

    best_smoothed_rewards_common = best_interp_func(common_timesteps)
    worst_smoothed_rewards_common = worst_interp_func(common_timesteps)

    # Fill the space between the best and worst smoothed rewards with 'aqua' color
    plt.fill_between(common_timesteps, worst_smoothed_rewards_common, best_smoothed_rewards_common, color='aqua')

    divide_sections_and_define_seeds(ax, common_timesteps, sections_number , chosen_seeds_best, chosen_seeds_worst)

    seeds_part = ",".join([f"{seed_name}={seed_value}" for seed_name, seed_value in seeds])
    graph_name = f"{algorithm}, {env_name},{seeds_part}, {parameter}={seed_range}"

    plt.title(graph_name, fontsize=16)
    plt.tight_layout()
    save_graph(plt, "best_model_vs_worst_model_name")

def extract_chosen_seeds_from_csv(csv_file_path, parameter):
    chosen_seeds = []

    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            seed_value = row['seed_value']
            chosen_seeds.append(f"{parameter}={seed_value}")

    return chosen_seeds

def extract_command_params(file_path: str):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        command = file.read().strip()

    # Regular expression pattern to match all parameters ending with _seed and their values
    seed_pattern = r'--(\w+_seed) (\d+)'
    seeds = re.findall(seed_pattern, command)

    # Regular expression patterns for sections_number, parameter, and seed_range
    sections_number_pattern = r'--sections_number (\d+)'
    parameter_pattern = r'--parameter (\w+)'
    seed_range_pattern = r'--seed_range (\d+,\d+)'
    algorithm_pattern = r'--algorithm (\w+)'
    env_name_pattern = r'--env_name (\w+)'

    # Extract the corresponding values
    sections_number = int(re.findall(sections_number_pattern, command)[0])
    parameter = re.findall(parameter_pattern, command)[0] + "_seed"
    algorithm = re.findall(algorithm_pattern, command)[0]
    env_name = re.findall(env_name_pattern, command)[0]

    seed_range = re.findall(seed_range_pattern, command)[0]
    start, end = seed_range.split(',')
    formatted_seed_range = f"[{start} ... {end}]"

    return algorithm, env_name, seeds, sections_number, parameter, formatted_seed_range

import os

def get_csv_path(directory):
    reward_logs_path = os.path.join(directory, LOG_DIR)
    # Find the first file in the reward_logs directory
    for file_name in os.listdir(reward_logs_path):
        if file_name:  # Ensures the file is not an empty string
            return os.path.join(reward_logs_path, file_name)
    raise FileNotFoundError(f"No files found in {reward_logs_path}")


def load_data_from_csv(model_name = None, path = LOG_DIR):
    """
    Loads the timestep and reward data from a CSV file for the specified model.

    :param model_name: The name of the model to search for in the CSV files.
    :return: A tuple of (timesteps, rewards) or (None, None) if the file is not found or is invalid.
    """
    if(model_name):
        model_name = model_name.replace(ZIP, "")
        search_pattern = os.path.join(path, f"*{model_name}*")
    else:
        search_pattern = path
        
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No CSV file found containing '{model_name}' in {LOG_DIR}.")
        return None, None

    csv_file = csv_files[0]
    data = pandas.read_csv(csv_file)

    if 'reward' not in data.columns or 'timestep' not in data.columns:
        print(f"CSV file {csv_file} does not contain 'reward' and 'timestep' columns.")
        return None, None

    return data['timestep'], data['reward']

def plot_graph(ax, timesteps, rewards, anti=False):
    """
    Plots the rewards over timesteps on the provided axes.

    :param ax: The.axes object to draw the plot on.
    :param timesteps: The timesteps data.
    :param rewards: The rewards data.
    :param sections_number: The number of vertical sections to split the graph into.
    :param anti: If True, change all graph colors to blue.
    :return: The smoothed rewards.
    """
    addition = "AntiSuperOracle" if anti else "SuperOracle"

    line_color = 'blue' if anti else 'deeppink'
    smoothed_color = 'navy' if anti else 'mediumvioletred'

    ax.plot(timesteps, rewards, label=f' {addition} Rewards', color=line_color, linestyle='-', marker='o', alpha=0.1)
    ax.fill_between(timesteps, rewards, color=line_color, alpha=0.1)
    
    smoothed_rewards = [numpy.mean(rewards[max(0, i-4):i+1]) for i in range(len(rewards))]
    ax.plot(timesteps, smoothed_rewards, label=f'{addition} Smoothed Rewards', color=smoothed_color, linestyle='-', linewidth=3, alpha=1.0)

    ax.set_xlim(left=0, right=max(timesteps)+200)
    ax.set_ylim(top=max(rewards)*1.5,bottom=min(rewards)*1.5)
    ax.set_xlabel('Learning Timesteps', fontsize=14)
    ax.set_ylabel('Rewards', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)

    return smoothed_rewards


def save_graph(plt, model_name, suffix=""):
    """
    Saves the current graph to a file.

    :param plt: The matplotlib plot object.
    :param model_name: The base name of the model to use for the file name.
    :param suffix: An optional suffix to append to the file name.
    """
    os.makedirs(GRAPH_DIR, exist_ok=True)
    graph_path = os.path.join(GRAPH_DIR, f"{model_name}{suffix}.png")
    plt.savefig(graph_path)
    plt.close()

    print(f"Graph saved to '{graph_path}'")

def draw_single_graph(model_name, sections_number, anti=False):
    """
    Draws and saves a single model's rewards over timesteps.

    :param model_name: The name of the model to plot.
    :param sections_number: The number of vertical sections to split the graph into.
    :param anti: If True, change all graph colors to blue.
    """
    seaborn.set(style="darkgrid")
    plt.figure(figsize=(13, 7))
    ax = plt.gca()  # Get current axes

    timesteps, rewards = load_data_from_csv(model_name)
    if timesteps is None:
        return

    plot_graph(ax, timesteps, rewards, anti)
    divide_sections_and_define_seeds(ax,timesteps,sections_number)
    
    plt.tight_layout()
    save_graph(plt, model_name)

