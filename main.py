# Main execution file for the project
import os
import pandas as pd
from request_to_download import download_file_from_github
import process_segmentation # leave

# Add argument to script to pass the path of the local data-multi-subject repository
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str)
args = parser.parse_args()


def main(dataset_dir):
    # Set dataset directory as global variable for other scripts in this project to use
    os.environ['DATASET_DIR'] = dataset_dir

    github_file_url = "https://raw.githubusercontent.com/spine-generic/data-multi-subject/master"
    participants_file_path = "participants.tsv"
    local_data_dir = "data"
    local_participants_path = os.path.join(local_data_dir, participants_file_path)

    os.makedirs(local_data_dir, exist_ok=True)
    download_file_from_github(os.path.join(github_file_url, participants_file_path), local_participants_path)

    # Not working yet
    # participants = pd.read_csv(local_participants_path, sep='\t')
    # local_t2w_dir = os.path.join(local_data_dir, "t2w")

    # for participant_id in participants['participant_id']:
    #     remote_file_path = f"{participant_id}/anat/{participant_id}_T2w.nii.gz"
    #     os.makedirs(local_t2w_dir, exist_ok=True)
    #     download_file_from_github(os.path.join(github_file_url, remote_file_path), 
    #                               os.path.join(local_t2w_dir, f"{participant_id}_T2w.nii.gz"))

    # # Initialize and retrieve files with git-annex
    # install_git_annex()
    # initialize_git_annex(local_t2w_dir)
    # retrieve_files_with_git_annex()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.dataset_dir)