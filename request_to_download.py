import requests
import subprocess
import os

def download_file_from_github(file_url, local_path):
    """
    Download a file from a GitHub repository.

    Parameters:
    - file_url (str): The full URL to the raw file in the GitHub repository.
    - local_path (str): The local path to save the downloaded file.
    """
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

# Example usage
# github_file_url = "https://raw.githubusercontent.com/spine-generic/data-multi-subject/master/participants.tsv"
# local_file_path = "data/participants.tsv"
# download_file_from_github(github_file_url, local_file_path)

def install_git_annex():
    # Example of installing git-annex (adjust based on your OS and package manager)
    subprocess.run(['sudo', 'apt-get', 'install', 'git-annex'], check=True)

def initialize_git_annex(directory):
    os.chdir(directory)
    subprocess.run(['git', 'annex', 'init'], check=True)

def retrieve_files_with_git_annex():
    subprocess.run(['git', 'annex', 'get', '.'], check=True)