import csv
import os
import shutil
import pandas

FIELDNAMES_BACKLOG = ["model_name", "average_reward"]
FIELDNAMES_REWARDS = ["reward","timestep"]
FIELDNAMES_RESULTS = [
    "anti",
    "algorithm",
    "section",
    "learning_timesteps",
    "environment",
    "env_seed",
    "policy_seed",
    "noise_seed",
    "buffer_seed",
    "parameter",
    "seed_value",
    "model_name",
    "avg_reward",
]
FILENAME_BACKLOG = "backlog_evaluation.csv"
FILENAME_RESULTS = "results.csv"
TEMP_MODELS_DIR = "tempModels"
BUFFER_DIR = "buffers"
NOISE_DIR = "noise"
LOG_DIR = "reward_logs"
GRAPH_DIR = "graph"
MODELS_DIR = "models"
SAC_SCRIPT = "SACscript.py"
SACname = "SAC"
PPOname = "PPO"
PPO_SCRIPT = "PPOscript.py"
ORACLE_SCRIPT = "oracle.py"
BUFFER = "_buffer"
NOISE = "_noise"
PKL = ".pkl"
ZIP = ".zip"
COMMAND = "command"

def write_to_csv(data, path, fieldnames=None,):
    """
    Writes data to a CSV file. Handles both dictionary and list data formats.

    Parameters:
    - data (dict or list): Data to be written to the CSV file.
    - filename (str): Name of the CSV file.
    - fieldnames (list, optional): List of fieldnames if data is a dictionary. Ignored if data is a list.
    """

    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as csv_file:
        if isinstance(data, dict):
            if fieldnames is None:
                raise ValueError("Fieldnames must be provided when writing dictionary data.")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        elif isinstance(data, list):
            writer = csv.writer(csv_file)
            if not file_exists and fieldnames is not None:
                writer.writerow(fieldnames)
            writer.writerow(data)
        else:
            raise TypeError("Data must be a dictionary or a list.")

def calculate_average_r(dir, model_name):
    """
    Calculates the average reward from a CSV file with 'reward' and 'timesteps' columns.
    
    :param folder: The folder where the CSV file is located.
    :param filename: The name of the CSV file containing the reward data.
    :return: The average reward.
    :raises FileNotFoundError: If the specified CSV file does not exist.
    """
    csv_file_path = os.path.join(dir, model_name)
    
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    try:
        data = pandas.read_csv(csv_file_path)
        
        return data['reward'].mean() if not data.empty else 0
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None

def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists.
    """
    os.makedirs(directory, exist_ok=True)

def copy_model(source_dir, model_name, suffix, extension):
    """
    Copy the best/worst model's file to the /models directory.
    """
    source_path = os.path.join(source_dir, model_name + suffix + extension)
    destination_path = os.path.join(MODELS_DIR, model_name + suffix + extension)
    shutil.copy(source_path, destination_path)

def clean_directory(directory, model_name):
    """
    Clean the specified directory by deleting all files and subdirectories within it,
    except for the best/worst model's file.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if model_name not in filename:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
def clean_files_in_folder_except_containing_name_in_current_section(dir, model_name):
    for filename in os.listdir(dir):
        if model_name not in filename:
            file_path = os.path.join(dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"Skipped (not a file): {file_path}")
        else:
            print(f"Kept: {filename}")

def clean_up(model_name):
    """
    Moves the best/worst-performing model, its buffer and noise dist to the '/models' directory
    and cleans the '/tempModels', '/buffers', '/noise' directories by deleting all other files.
    """
    ensure_directory_exists(MODELS_DIR)

    # Handle models
    ensure_directory_exists(TEMP_MODELS_DIR)
    copy_model(TEMP_MODELS_DIR, model_name, "", ZIP)
    clean_directory(TEMP_MODELS_DIR, model_name)

    # Handle buffers
    ensure_directory_exists(BUFFER_DIR)
    copy_model(BUFFER_DIR, model_name, BUFFER, PKL)
    clean_directory(BUFFER_DIR, model_name)

    # Handle noise
    ensure_directory_exists(NOISE_DIR)
    copy_model(NOISE_DIR, model_name, NOISE, PKL)
    clean_directory(NOISE_DIR, model_name)

    clean_files_in_folder_except_containing_name_in_current_section(LOG_DIR,model_name)

def find_model_with_name_contains(directory, search_string):
    """
    Searches for a file in the specified directory whose name contains the given search string and ends with ".zip".
    Raises an error if multiple or no matches are found.
    """
    matches = []
    for filename in os.listdir(directory):
        if search_string in filename and filename.endswith(ZIP):
            matches.append(filename)
    
    if len(matches) > 1:
        raise ValueError(f'Multiple models found: {matches}')
    elif len(matches) == 1:
        print(f'Model found: { matches[0]}')
        return matches[0]
    else:
        raise FileNotFoundError(f"Model does not exist. Serach string : {search_string}")
    
def delete_folder(folder_name):
    """
    Delete a folder and all its contents.
    """
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        try:
            shutil.rmtree(folder_name)
            print(f"Successfully deleted the folder: {folder_name}")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")
    else:
        print(f"The folder '{folder_name}' does not exist or is not a directory.")

def manage_model_log(dir, model_name, input_model_name):
    path = os.path.join(dir, model_name)

    if(input_model_name):
        input_model_name = input_model_name.replace(ZIP, "")
        input_file_path = os.path.join(dir, f"{input_model_name}")
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"File {input_file_path} not found.")
        
        output_file_path = os.path.join(dir, f"{model_name}")
        shutil.copyfile(input_file_path, output_file_path)
    else:
        try:
            os.makedirs(dir, exist_ok=True)
        except FileExistsError:
            pass
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES_REWARDS)
            writer.writeheader()

    # Append an empty line at the end of the file
    with open(path, 'a', newline='') as f:
        f.write('\n')
            
    return path