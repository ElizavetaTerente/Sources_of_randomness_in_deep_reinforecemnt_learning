from common import (
    MODELS_DIR,
    TEMP_MODELS_DIR,
    NOISE_DIR,
    BUFFER_DIR,
    LOG_DIR,
    GRAPH_DIR,
    FILENAME_BACKLOG,
    FILENAME_RESULTS,
    COMMAND
)

import os
import shutil

# List of directories to remove
directories_to_remove = [
    MODELS_DIR,
    TEMP_MODELS_DIR,
    NOISE_DIR,
    BUFFER_DIR,
    LOG_DIR,
    GRAPH_DIR
]

# List of files to remove
files_to_remove = [
    FILENAME_BACKLOG,
    FILENAME_RESULTS,
    COMMAND
]

def remove_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Directory {directory} removed successfully.")
            except Exception as e:
                print(f"Error removing directory {directory}: {e}")
        else:
            print(f"Directory {directory} does not exist.")

def remove_files(files):
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"File {file} removed successfully.")
            except Exception as e:
                print(f"Error removing file {file}: {e}")
        else:
            print(f"File {file} does not exist.")

if __name__ == "__main__":
    remove_directories(directories_to_remove)
    remove_files(files_to_remove)



