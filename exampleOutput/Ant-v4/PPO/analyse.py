import os
import numpy as np
import pandas as pd
from graph import (
    SUPER,
    ANTI,
    get_csv_path
)

def load_data_and_split_into_sections(path):
    """
    Loads the data from a CSV file and splits it into sections based on blank lines.
    Each section corresponds to a list of tuples (reward, timestep).
    """
    try:
        sections = []
        current_section = []
        
        with open(path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                if line.strip() == "":
                    if current_section:
                        sections.append(current_section)
                        current_section = []
                else:
                    # Parse the line into reward and timestep
                    reward, timestep = map(float, line.strip().split(','))
                    current_section.append((reward, timestep))
                    
            # Add the last section if it exists
            if current_section:
                sections.append(current_section)
        
        return sections
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None

def calculate_smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two lists of values,
    handling potential issues with small or negative y_true values.
    
    :param y_true: List of true values (best performance).
    :param y_pred: List of predicted values (worst performance).
    :return: SMAPE value as a float.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100


def calculate_smape_and_save(source: str, path_to_super=SUPER, path_to_anti=ANTI, output_filename="smape_results.csv"):
    """
    Calculate the MAPE between rewards from 'super' and 'anti' directories per section,
    and save the results to a CSV file in the root directory. Also, calculate the MAPE
    across all sections combined and add it as a separate entry in the CSV file.
    
    The results will be saved with an additional 'source' column.
    If the output file already exists, new results will be appended.
    
    :param source: A string representing the source of the data, to be included in the CSV.
    :param path_to_super: Path to the directory containing the 'super' CSV files.
    :param path_to_anti: Path to the directory containing the 'anti' CSV files.
    :param output_filename: The name of the output CSV file.
    """
    super_csv_path = get_csv_path(path_to_super)
    anti_csv_path = get_csv_path(path_to_anti)
    
    if super_csv_path is None or anti_csv_path is None:
        print("CSV files not found in the specified folders.")
        return
    
    super_sections = load_data_and_split_into_sections(super_csv_path)
    anti_sections = load_data_and_split_into_sections(anti_csv_path)
    
    if super_sections is None or anti_sections is None:
        print("Failed to load or split reward logs.")
        return
    
    num_sections = min(len(super_sections), len(anti_sections))
    super_sections = super_sections[:num_sections]
    anti_sections = anti_sections[:num_sections]
    
    smape_results = []
    all_rewards1 = []
    all_rewards2 = []

    try:
        for s1, s2 in zip(super_sections, anti_sections):
            smape = calculate_smape(s1, s2)
            smape_results.append(smape)
            all_rewards1.extend(s1)
            all_rewards2.extend(s2)

        # Calculate MAPE across all sections combined
        smape_all = calculate_smape(all_rewards1, all_rewards2)
        
    except ValueError as e:
        print(f"Error calculating SMAPE: {e}")
        return
    
    # Create a DataFrame to save the results
    if num_sections==1:
        mape_df = pd.DataFrame({
            "source": [source],
            "Section": ["all"],
            "SMAPE":  [smape_all]
        })
    else:
                mape_df = pd.DataFrame({
            "source": [source] * (num_sections + 1),
            "Section": list(range(1, num_sections + 1)) + ["all"],
            "SMAPE": smape_results + [smape_all]
        })
    
    output_path = os.path.join(os.getcwd(), output_filename)
    
    # If the file exists, append; otherwise, create a new file
    if os.path.exists(output_path):
        mape_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        mape_df.to_csv(output_path, index=False)
    
    print(f"SMAPE results saved to {output_path}")

# def format_logs(folder_path):
#     # Define the paths for the two folders
#     super_folder = os.path.join(folder_path, "super", "reward_logs")
#     anti_folder = os.path.join(folder_path, "anti", "reward_logs")
    
#     # Find the single file in each reward_logs folder
#     super_log_file = [f for f in os.listdir(super_folder) if os.path.isfile(os.path.join(super_folder, f))][0]
#     anti_log_file = [f for f in os.listdir(anti_folder) if os.path.isfile(os.path.join(anti_folder, f))][0]
    
#     super_df = pd.read_csv(os.path.join(super_folder, super_log_file))
#     anti_df = pd.read_csv(os.path.join(anti_folder, anti_log_file))
    
#     # Rename columns to avoid confusion
#     super_df.columns = ['reward', 'timestep']
#     anti_df.columns = ['reward', 'timestep']
    
#     # Synchronize super_df with timesteps from anti_df
#     combined_timesteps = pd.Series(sorted(set(super_df['timestep']).union(set(anti_df['timestep']))))
#     synchronized_super_df = pd.merge_asof(combined_timesteps.to_frame(name='timestep'), super_df, on='timestep', direction='backward')
    
#     # Fill in missing rewards using forward fill, then backward fill as fallback
#     synchronized_super_df['reward'].fillna(method='ffill', inplace=True)
#     synchronized_super_df['reward'].fillna(method='bfill', inplace=True)
    
#     # Synchronize anti_df with timesteps from super_df
#     synchronized_anti_df = pd.merge_asof(combined_timesteps.to_frame(name='timestep'), anti_df, on='timestep', direction='backward')
    
#     # Fill in missing rewards using forward fill, then backward fill as fallback
#     synchronized_anti_df['reward'].fillna(method='ffill', inplace=True)
#     synchronized_anti_df['reward'].fillna(method='bfill', inplace=True)
    
#     # Save the synchronized dataframes to new files with "_formatted" appended to their names
#     synchronized_super_path = os.path.join(super_folder, f"{super_log_file}_formatted.csv")
#     synchronized_anti_path = os.path.join(anti_folder, f"{anti_log_file}_formatted.csv")
    
#     synchronized_super_df.to_csv(synchronized_super_path, index=False)
#     synchronized_anti_df.to_csv(synchronized_anti_path, index=False)
    
#     print(f"Synchronized files created: \n{os.path.basename(synchronized_super_path)}\n{os.path.basename(synchronized_anti_path)}")

# import os
# import pandas as pd

# def split_into_sections(file_path):
#     sections = []
#     current_section = []
    
#     with open(file_path, 'r') as file:
#         for line in file:
#             stripped_line = line.strip()
#             if not stripped_line:  # Empty line indicates the start of a new section
#                 if current_section:
#                     sections.append(current_section)
#                     current_section = []
#             else:
#                 current_section.append(stripped_line)
#         if current_section:  # Append the last section if file doesn't end with an empty line
#             sections.append(current_section)
    
#     return sections

# def write_sections_to_files(sections, original_file_path):
#     base_name = os.path.splitext(original_file_path)[0]
#     section_files = []
    
#     for i, section in enumerate(sections, start=0):
#         if i==0:
#             continue  # Skip the iteration when i == 0
#         section_file_path = f"{base_name}_section_{i}.csv"
#         with open(section_file_path, 'w') as section_file:
#             section_file.write('\n'.join(section) + '\n')
#         section_files.append(section_file_path)
    
#     return section_files

# def format_logs_in_section(section_file_path, combined_timesteps):
#     df = pd.read_csv(section_file_path, header=None, names=['reward', 'timestep'])
    
#     # Convert the timestep column to numeric to avoid merge issues
#     df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
#     combined_timesteps = pd.to_numeric(combined_timesteps, errors='coerce')
    
#     # Synchronize data
#     synchronized_df = pd.merge_asof(combined_timesteps.to_frame(name='timestep'), df, on='timestep', direction='backward')
    
#     # Fill in missing rewards using forward fill, then backward fill as fallback
#     synchronized_df['reward'].fillna(method='ffill', inplace=True)
#     synchronized_df['reward'].fillna(method='bfill', inplace=True)
    
#     # Save the synchronized dataframe to the same section file
#     synchronized_df.to_csv(section_file_path, index=False, header=False)

# def combine_sections_to_result(section_files, result_file_path):
#     with open(result_file_path, 'w') as result_file:
#         for section_file in section_files:
#             with open(section_file, 'r') as file:
#                 result_file.write(file.read())
#             result_file.write('\n')  # Add empty line to separate sections
    
#     # Cleanup section files
#     for section_file in section_files:
#         os.remove(section_file)

# def process_log_files(folder_path):
#     super_folder = os.path.join(folder_path, "super", "reward_logs")
#     anti_folder = os.path.join(folder_path, "anti", "reward_logs")

#     # Identify the log files in each folder
#     super_log_file = [f for f in os.listdir(super_folder) if os.path.isfile(os.path.join(super_folder, f))][0]
#     anti_log_file = [f for f in os.listdir(anti_folder) if os.path.isfile(os.path.join(anti_folder, f))][0]

#     super_file_path = os.path.join(super_folder, super_log_file)
#     anti_file_path = os.path.join(anti_folder, anti_log_file)

#     # Step 1: Split the original files into sections
#     super_sections = split_into_sections(super_file_path)
#     anti_sections = split_into_sections(anti_file_path)

#     # Step 2: Write each section to a separate file
#     super_section_files = write_sections_to_files(super_sections, super_file_path)
#     anti_section_files = write_sections_to_files(anti_sections, anti_file_path)

#     # Step 3: Process each section separately
#     for super_section_file, anti_section_file in zip(super_section_files, anti_section_files):
#         # Read timesteps from both files to create a combined timeline
#         super_df = pd.read_csv(super_section_file, header=None, names=['reward', 'timestep'])
#         anti_df = pd.read_csv(anti_section_file, header=None, names=['reward', 'timestep'])

#         # Ensure timesteps are numeric
#         super_df['timestep'] = pd.to_numeric(super_df['timestep'], errors='coerce')
#         anti_df['timestep'] = pd.to_numeric(anti_df['timestep'], errors='coerce')

#         combined_timesteps = pd.Series(sorted(set(super_df['timestep']).union(set(anti_df['timestep']))))

#         format_logs_in_section(super_section_file, combined_timesteps)
#         format_logs_in_section(anti_section_file, combined_timesteps)
    
#     # Step 4: Combine all processed sections into the final result file
#     super_result_file_path = os.path.join(super_folder, f"{super_log_file}_formatted.csv")
#     anti_result_file_path = os.path.join(anti_folder, f"{anti_log_file}_formatted.csv")

#     combine_sections_to_result(super_section_files, super_result_file_path)
#     combine_sections_to_result(anti_section_files, anti_result_file_path)
    
#     print(f"Resulting files created:\n{super_result_file_path}\n{anti_result_file_path}")

# import os
# import pandas as pd

# def split_into_sections(file_path):
#     sections = []
#     current_section = []
    
#     with open(file_path, 'r') as file:
#         header = file.readline().strip()  # Read the header
#         for line in file:
#             stripped_line = line.strip()
#             if not stripped_line:  # Empty line indicates the start of a new section
#                 if current_section:
#                     sections.append(current_section)
#                     current_section = []
#             else:
#                 current_section.append(stripped_line)
#         if current_section:  # Append the last section if file doesn't end with an empty line
#             sections.append(current_section)
    
#     return sections, header

# def write_sections_to_files(sections, original_file_path, header):
#     base_name = os.path.splitext(original_file_path)[0]
#     section_files = []
    
#     for i, section in enumerate(sections, start=0):
#         section_file_path = f"{base_name}_section_{i}.csv"
#         with open(section_file_path, 'w') as section_file:
#             section_file.write(header + '\n')  # Write the header
#             section_file.write('\n'.join(section) + '\n')
#         section_files.append(section_file_path)
    
#     return section_files

# def format_logs_in_section(section_file_path, combined_timesteps):
#     df = pd.read_csv(section_file_path)
    
#     # Convert the timestep column to numeric to avoid merge issues
#     df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
#     combined_timesteps = pd.to_numeric(combined_timesteps, errors='coerce')
    
#     # Synchronize data
#     synchronized_df = pd.merge_asof(combined_timesteps.to_frame(name='timestep'), df, on='timestep', direction='backward')
    
#     # Fill in missing rewards using forward fill, then backward fill as fallback
#     synchronized_df['reward'].fillna(method='ffill', inplace=True)
#     synchronized_df['reward'].fillna(method='bfill', inplace=True)
    
#     # Save the synchronized dataframe to the same section file
#     synchronized_df.to_csv(section_file_path, index=False)

# def combine_sections_to_result(section_files, result_file_path):
#     with open(result_file_path, 'w') as result_file:
#         for i, section_file in enumerate(section_files):
#             with open(section_file, 'r') as file:
#                 if i == 0:  # Write the header only once
#                     result_file.write(file.readline())  # Write the header from the first section
#                 else:
#                     file.readline()  # Skip the header in subsequent sections
#                 result_file.write(file.read())
#             result_file.write('\n')  # Add empty line to separate sections
    
#     # Cleanup section files
#     for section_file in section_files:
#         os.remove(section_file)

# def process_log_files(folder_path):
#     super_folder = os.path.join(folder_path, "super", "reward_logs")
#     anti_folder = os.path.join(folder_path, "anti", "reward_logs")

#     # Identify the log files in each folder
#     super_log_file = [f for f in os.listdir(super_folder) if os.path.isfile(os.path.join(super_folder, f))][0]
#     anti_log_file = [f for f in os.listdir(anti_folder) if os.path.isfile(os.path.join(anti_folder, f))][0]

#     super_file_path = os.path.join(super_folder, super_log_file)
#     anti_file_path = os.path.join(anti_folder, anti_log_file)

#     # Step 1: Split the original files into sections and get headers
#     super_sections, super_header = split_into_sections(super_file_path)
#     anti_sections, anti_header = split_into_sections(anti_file_path)

#     # Step 2: Write each section to a separate file
#     super_section_files = write_sections_to_files(super_sections, super_file_path, super_header)
#     anti_section_files = write_sections_to_files(anti_sections, anti_file_path, anti_header)

#     # Step 3: Process each section separately
#     for super_section_file, anti_section_file in zip(super_section_files, anti_section_files):
#         # Read timesteps from both files to create a combined timeline
#         super_df = pd.read_csv(super_section_file)
#         anti_df = pd.read_csv(anti_section_file)

#         # Ensure timesteps are numeric
#         super_df['timestep'] = pd.to_numeric(super_df['timestep'], errors='coerce')
#         anti_df['timestep'] = pd.to_numeric(anti_df['timestep'], errors='coerce')

#         combined_timesteps = pd.Series(sorted(set(super_df['timestep']).union(set(anti_df['timestep']))))

#         format_logs_in_section(super_section_file, combined_timesteps)
#         format_logs_in_section(anti_section_file, combined_timesteps)
    
#     # Step 4: Combine all processed sections into the final result file
#     super_result_file_path = os.path.join(super_folder, f"{super_log_file}_formatted.csv")
#     anti_result_file_path = os.path.join(anti_folder, f"{anti_log_file}_formatted.csv")

#     combine_sections_to_result(super_section_files, super_result_file_path)
#     combine_sections_to_result(anti_section_files, anti_result_file_path)
    
#     print(f"Resulting files created:\n{super_result_file_path}\n{anti_result_file_path}")

import os
import pandas as pd

def split_into_sections(file_path):
    sections = []
    current_section = []
    
    with open(file_path, 'r') as file:
        header = file.readline().strip()  # Read the header
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:  # Empty line indicates the start of a new section
                if current_section:
                    sections.append(current_section)
                    current_section = []
            else:
                current_section.append(stripped_line)
        if current_section:  # Append the last section if file doesn't end with an empty line
            sections.append(current_section)
    
    return sections, header

def write_sections_to_files(sections, original_file_path, header):
    base_name = os.path.splitext(original_file_path)[0]
    section_files = []
    
    for i, section in enumerate(sections, start=0):
        section_file_path = f"{base_name}_section_{i}.csv"
        with open(section_file_path, 'w') as section_file:
            section_file.write(header + '\n')  # Write the header
            section_file.write('\n'.join(section) + '\n')
        section_files.append(section_file_path)
    
    return section_files

def format_logs_in_section(section_file_path, combined_timesteps):
    df = pd.read_csv(section_file_path)
    
    # Convert the timestep column to numeric to avoid merge issues
    df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
    combined_timesteps = pd.to_numeric(combined_timesteps, errors='coerce')
    
    # Synchronize data
    synchronized_df = pd.merge_asof(combined_timesteps.to_frame(name='timestep'), df, on='timestep', direction='backward')
    
    # Fill in missing rewards using forward fill, then backward fill as fallback
    synchronized_df['reward'].fillna(method='ffill', inplace=True)
    synchronized_df['reward'].fillna(method='bfill', inplace=True)
    
    # Save the synchronized dataframe to the same section file
    synchronized_df.to_csv(section_file_path, index=False)

def combine_sections_to_result(section_files, result_file_path):
    with open(result_file_path, 'w') as result_file:
        for i, section_file in enumerate(section_files):
            with open(section_file, 'r') as file:
                if i == 0:  # Write the header only once
                    result_file.write(file.readline())  # Write the header from the first section
                else:
                    file.readline()  # Skip the header in subsequent sections
                result_file.write(file.read())
            result_file.write('\n')  # Add empty line to separate sections
    
    # Cleanup section files
    for section_file in section_files:
        os.remove(section_file)

def rename_original_file(original_file_path):
    unformatted_file_path = f"{os.path.splitext(original_file_path)[0]}_unformatted.csv"
    os.rename(original_file_path, unformatted_file_path)
    return unformatted_file_path

def process_log_files(folder_path):
    super_folder = os.path.join(folder_path, "super", "reward_logs")
    anti_folder = os.path.join(folder_path, "anti", "reward_logs")

    # Identify the log files in each folder
    super_log_file = [f for f in os.listdir(super_folder) if os.path.isfile(os.path.join(super_folder, f))][0]
    anti_log_file = [f for f in os.listdir(anti_folder) if os.path.isfile(os.path.join(anti_folder, f))][0]

    super_file_path = os.path.join(super_folder, super_log_file)
    anti_file_path = os.path.join(anti_folder, anti_log_file)

    # Rename original files
    super_unformatted_file_path = rename_original_file(super_file_path)
    anti_unformatted_file_path = rename_original_file(anti_file_path)

    # Step 1: Split the original files into sections and get headers
    super_sections, super_header = split_into_sections(super_unformatted_file_path)
    anti_sections, anti_header = split_into_sections(anti_unformatted_file_path)

    # Step 2: Write each section to a separate file
    super_section_files = write_sections_to_files(super_sections, super_unformatted_file_path, super_header)
    anti_section_files = write_sections_to_files(anti_sections, anti_unformatted_file_path, anti_header)

    # Step 3: Process each section separately
    for super_section_file, anti_section_file in zip(super_section_files, anti_section_files):
        # Read timesteps from both files to create a combined timeline
        super_df = pd.read_csv(super_section_file)
        anti_df = pd.read_csv(anti_section_file)

        # Ensure timesteps are numeric
        super_df['timestep'] = pd.to_numeric(super_df['timestep'], errors='coerce')
        anti_df['timestep'] = pd.to_numeric(anti_df['timestep'], errors='coerce')

        combined_timesteps = pd.Series(sorted(set(super_df['timestep']).union(set(anti_df['timestep']))))

        format_logs_in_section(super_section_file, combined_timesteps)
        format_logs_in_section(anti_section_file, combined_timesteps)
    
    # Step 4: Combine all processed sections into the final result file
    combine_sections_to_result(super_section_files, super_file_path)  # Use original file name for formatted file
    combine_sections_to_result(anti_section_files, anti_file_path)    # Use original file name for formatted file
    
    print(f"Resulting files created:\n{super_file_path}\n{anti_file_path}")
    print(f"Original unformatted files:\n{super_unformatted_file_path}\n{anti_unformatted_file_path}")



