import logging
from datetime import datetime
from typing import List
from pathlib import Path
def check_and_create_directories(paths_to_check: List[Path]):
    """
    Creates all directories needed for the specified paths
    :param paths_to_check: All the paths for which existence should be checked
    :return:
    """
    for path in paths_to_check:
        dir_path = path.parent
        # Create directories if they don't exist
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    print("All directories correctly setup.")
    return

def set_up_logging(log_file_name: str):
    log_dir = Path("../../logs")
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    log_filename = log_dir / str(log_file_name + f"_{datetime.now().strftime('%Y-%m-%d')}")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )