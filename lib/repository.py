import os
import subprocess
import sys
import shutil

def download_github_repo(repo_url, destination_folder):
    # Check if the repository already exists
    if os.path.exists(destination_folder):
        print(f"Repository already exists at {destination_folder}. Skipping download.")
        return

    # Ensure git is installed
    if not shutil.which("git"):
        print("Git is not installed. Please install Git and try again.")
        sys.exit(1)

    # Ensure the destination folder exists
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    # Construct the git clone command
    command = ["git", "clone", repo_url, destination_folder]

    try:
        # Execute the git clone command
        subprocess.run(command, check=True)
        print(f"Repository cloned into {destination_folder}")
        init_git_command = ["cd", destination_folder, "&&", "git", "init"]
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cloning the repository: {e}")
        sys.exit(1)
