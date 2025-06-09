import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def estimate_optimal_seed(with_graphs=False):
    # Load the CSV file
    file_path = 'backlog_evaluation.csv'
    df = pd.read_csv(file_path)

    # Function to extract the highest seed number from the model_name column
    def extract_max_seeds(df):
        seed_numbers = []
        pattern = re.compile(r"=(\d+)")
        
        for model_name in df['model_name']:
            matches = pattern.findall(model_name)
            if matches:
                seed_numbers.extend([int(match) for match in matches])
        
        if seed_numbers:
            return max(seed_numbers)
        else:
            return 1

    # Determine max_seeds
    max_seeds = extract_max_seeds(df)
    max_seeds = min(max_seeds, len(df))  # Ensure max_seeds does not exceed the number of records

    # Method 1: Captured Average Reward Difference
    def calculate_captured_difference(n, df):
        differences = []
        for _ in range(100):
            sampled_rewards = df['average_reward'].sample(n, random_state=np.random.randint(1, 10000))
            difference = sampled_rewards.max() - sampled_rewards.min()
            differences.append(difference)
        return np.mean(differences)

    captured_differences = [calculate_captured_difference(n, df) for n in range(1, max_seeds + 1)]

    stabilization_threshold = 0.1
    stable_seed_number_reward_difference = next(
        (i for i in range(1, len(captured_differences)) 
        if abs(captured_differences[i] - captured_differences[i-1]) < stabilization_threshold), 
        max_seeds
    )

    # Method 2: Bootstrap Method
    def calculate_captured_difference_bootstrap(n, df):
        differences = []
        for _ in range(100):
            sampled_rewards = df['average_reward'].sample(n, replace=True)
            difference = sampled_rewards.max() - sampled_rewards.min()
            differences.append(difference)
        return np.mean(differences)

    captured_differences_bootstrap = [calculate_captured_difference_bootstrap(n, df) for n in range(1, max_seeds + 1)]

    stable_seed_number_bootstrap = next(
        (i for i in range(1, len(captured_differences_bootstrap)) 
        if abs(captured_differences_bootstrap[i] - captured_differences_bootstrap[i-1]) < stabilization_threshold), 
        max_seeds
    )

    # Method 3: Moving Average Method
    window_size = 5
    moving_averages = np.convolve(captured_differences, np.ones(window_size)/window_size, mode='valid')

    stable_seed_number_moving_avg = next(
        (i for i in range(1, len(moving_averages)) 
        if abs(moving_averages[i] - moving_averages[i-1]) < stabilization_threshold), 
        max_seeds - window_size + 1
    )

    if stable_seed_number_moving_avg:
        stable_seed_number_moving_avg = min(stable_seed_number_moving_avg + window_size, max_seeds)  # Adjust for window size but ensure within max_seeds

    # Method 4: Cumulative Mean and Standard Deviation Method
    def calculate_cumulative_mean_std(df):
        cumulative_means = []
        cumulative_stds = []
        
        for n in range(1, max_seeds + 1):
            sampled_rewards = df['average_reward'].sample(n, random_state=np.random.randint(1, 10000))
            cumulative_means.append(sampled_rewards.mean())
            cumulative_stds.append(sampled_rewards.std())
        
        return cumulative_means, cumulative_stds

    cumulative_means, cumulative_stds = calculate_cumulative_mean_std(df)

    stable_seed_number_cumulative = next(
        (i for i in range(1, len(cumulative_means)) 
        if abs(cumulative_means[i] - cumulative_means[i-1]) < stabilization_threshold), 
        max_seeds
    )

    # Print the results of all four methods
    print(f"(Reward Difference Method) The difference stabilizes around {stable_seed_number_reward_difference} seeds.")
    print(f"(Bootstrap Method) The difference stabilizes around {stable_seed_number_bootstrap} seeds.")
    print(f"(Moving Average Method) The difference stabilizes around {stable_seed_number_moving_avg} seeds.")
    print(f"(Cumulative Mean and Std Dev Method) The difference stabilizes around {stable_seed_number_cumulative} seeds.")

    # Aggregate the results
    stabilization_points = [
        stable_seed_number_reward_difference, 
        stable_seed_number_bootstrap, 
        stable_seed_number_moving_avg, 
        stable_seed_number_cumulative
    ]

    # Determine the optimal number of seeds based on all four methods
    # Here we use the median to balance the influence of each method
    optimal_seed_number = int(np.median(stabilization_points))


    print(f"The optimal number of seeds based on all four methods is {optimal_seed_number}.")

    # Save the plots and results
    # if(with_graphs):

        # output_file_path_reward_difference = 'captured_reward_difference_plot_performance_range.png'
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, max_seeds + 1), captured_differences, marker='o')
    #     plt.axvline(x=stable_seed_number_reward_difference, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_reward_difference}')
    #     plt.xlabel('Number of Seeds (n)')
    #     plt.ylabel('Captured Average Reward Difference')
    #     plt.title('Captured Average Reward Difference vs. Number of Seeds (Performance Range Method)')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(output_file_path_reward_difference)

    #     output_file_path_bootstrap = 'captured_reward_difference_plot_bootstrap.png'
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, max_seeds + 1), captured_differences_bootstrap, marker='o')
    #     plt.axvline(x=stable_seed_number_bootstrap, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_bootstrap}')
    #     plt.xlabel('Number of Seeds (n)')
    #     plt.ylabel('Captured Average Reward Difference')
    #     plt.title('Captured Average Reward Difference vs. Number of Seeds (Bootstrap Method)')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(output_file_path_bootstrap)

    #     output_file_path_moving_avg = 'captured_reward_difference_plot_moving_avg.png'
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, max_seeds + 1), captured_differences, marker='o', label='Captured Differences')
    #     plt.plot(range(window_size, max_seeds + 1), moving_averages, color='red', linestyle='dashed', label=f'{window_size}-Point Moving Average')
    #     plt.axvline(x=stable_seed_number_moving_avg, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_moving_avg}')
    #     plt.xlabel('Number of Seeds (n)')
    #     plt.ylabel('Captured Average Reward Difference')
    #     plt.title('Captured Average Reward Difference vs. Number of Seeds (Moving Average Method)')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(output_file_path_moving_avg)

    #     output_file_path_cumulative = 'captured_reward_difference_plot_cumulative.png'
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, max_seeds + 1), cumulative_means, marker='o', label='Cumulative Mean')
    #     plt.fill_between(range(1, max_seeds + 1), 
    #                     np.array(cumulative_means) - np.array(cumulative_stds), 
    #                     np.array(cumulative_means) + np.array(cumulative_stds), 
    #                     color='b', alpha=0.2, label='Mean ± 1 Std Dev')
    #     plt.axvline(x=stable_seed_number_cumulative, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_cumulative}')
    #     plt.xlabel('Number of Seeds (n)')
    #     plt.ylabel('Cumulative Mean ± Std Dev')
    #     plt.title('Cumulative Mean and Std Dev vs. Number of Seeds')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(output_file_path_cumulative)



    #     print(f"Plots saved as {output_file_path_reward_difference}, {output_file_path_bootstrap}, {output_file_path_moving_avg}, and {output_file_path_cumulative}.")

    if(with_graphs):
        output_file_path_reward_difference = 'captured_reward_difference_plot_performance_range.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_seeds + 1), captured_differences, marker='o')
        plt.axvline(x=stable_seed_number_reward_difference, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_reward_difference}')
        plt.xlabel('Number of Seeds (n)', fontsize=20)  # Increase fontsize
        plt.ylabel('Captured Average Reward Difference', fontsize=20)  # Increase fontsize
        plt.title('Captured Average Reward Difference vs. Number of Seeds (Performance Range Method)', fontsize=13)  # Increase fontsize
        plt.grid(True)
        plt.legend(fontsize=16)  # Increase legend fontsize
        plt.savefig(output_file_path_reward_difference)

        output_file_path_bootstrap = 'captured_reward_difference_plot_bootstrap.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_seeds + 1), captured_differences_bootstrap, marker='o')
        plt.axvline(x=stable_seed_number_bootstrap, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_bootstrap}')
        plt.xlabel('Number of Seeds (n)', fontsize=20)  # Increase fontsize
        plt.ylabel('Captured Average Reward Difference', fontsize=20)  # Increase fontsize
        plt.title('Captured Average Reward Difference vs. Number of Seeds (Bootstrap Method)', fontsize=13)  # Increase fontsize
        plt.grid(True)
        plt.legend(fontsize=16)  # Increase legend fontsize
        plt.savefig(output_file_path_bootstrap)


        output_file_path_moving_avg = 'captured_reward_difference_plot_moving_avg.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_seeds + 1), captured_differences, marker='o', label='Captured Differences')
        plt.plot(range(window_size, max_seeds + 1), moving_averages, color='red', linestyle='dashed', label=f'{window_size}-Point Moving Average')
        plt.axvline(x=stable_seed_number_moving_avg, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_moving_avg}')
        plt.xlabel('Number of Seeds (n)', fontsize=20)  # Increase fontsize
        plt.ylabel('Captured Average Reward Difference', fontsize=20)  # Increase fontsize
        plt.title('Captured Average Reward Difference vs. Number of Seeds (Moving Average Method)', fontsize=13)  # Increase fontsize
        plt.grid(True)
        plt.legend(fontsize=16)  # Increase legend fontsize
        plt.savefig(output_file_path_moving_avg)

        output_file_path_cumulative = 'captured_reward_difference_plot_cumulative.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_seeds + 1), cumulative_means, marker='o', label='Cumulative Mean')
        plt.fill_between(range(1, max_seeds + 1), 
                        np.array(cumulative_means) - np.array(cumulative_stds), 
                        np.array(cumulative_means) + np.array(cumulative_stds), 
                        color='b', alpha=0.2, label='Mean ± 1 Std Dev')
        plt.axvline(x=stable_seed_number_cumulative, color='red', linestyle='--', label=f'Stable Seed Range = {stable_seed_number_cumulative}')
        plt.xlabel('Number of Seeds (n)', fontsize=20)  # Increase fontsize
        plt.ylabel('Cumulative Mean ± Std Dev', fontsize=20)  # Increase fontsize
        plt.title('Cumulative Mean and Std Dev vs. Number of Seeds', fontsize=13)  # Increase fontsize
        plt.grid(True)
        plt.legend(fontsize=16)  # Increase legend fontsize
        plt.savefig(output_file_path_cumulative)

        print(f"Plots saved as {output_file_path_reward_difference}, {output_file_path_bootstrap}, {output_file_path_moving_avg}, and {output_file_path_cumulative}.")


    return optimal_seed_number,stable_seed_number_reward_difference,stable_seed_number_bootstrap,stable_seed_number_moving_avg,stable_seed_number_cumulative


def perform_estimation(n_runs=100):
    optimal_seeds = []
    reward_difference_seeds = []
    bootstrap_seeds = []
    moving_avg_seeds = []
    cumulative_seeds = []

    for i in range(n_runs):
        print(f"Running estimation {i + 1}/{n_runs}...")
        optimal_seed, reward_difference_seed, bootstrap_seed, moving_avg_seed, cumulative_seed = estimate_optimal_seed(with_graphs=True)
        optimal_seeds.append(optimal_seed)
        reward_difference_seeds.append(reward_difference_seed)
        bootstrap_seeds.append(bootstrap_seed)
        moving_avg_seeds.append(moving_avg_seed)
        cumulative_seeds.append(cumulative_seed)

    # Calculate the final optimal seed based on the median of all runs
    final_optimal_seed = int(np.median(optimal_seeds))
    final_reward_difference_seed = int(np.median(reward_difference_seeds))
    final_bootstrap_seed = int(np.median(bootstrap_seeds))
    final_moving_avg_seed = int(np.median(moving_avg_seeds))
    final_cumulative_seed = int(np.median(cumulative_seeds))

    print(f"Final median is {np.median([final_reward_difference_seed,final_bootstrap_seed,final_moving_avg_seed,final_cumulative_seed])}")
    print(f"Final Reward Difference Method seeds number after {n_runs} runs is {final_reward_difference_seed}.")
    print(f"Final Moving Average Method seeds number after {n_runs} runs is {final_moving_avg_seed}.")
    print(f"Final Bootstrap Method seeds number after {n_runs} runs is {final_bootstrap_seed}.")
    print(f"Final Cumulative Mean and Std Dev Method seeds number after {n_runs} runs is {final_cumulative_seed}.")
    print(f"Final optimal seeds number after {n_runs} runs : {final_optimal_seed}.")

if __name__ == "__main__":
    n_runs = 100  # You can set the number of runs here or take input
    final_optimal_seed = perform_estimation(n_runs=n_runs)

