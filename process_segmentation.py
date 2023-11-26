import os
import pandas as pd
import subprocess
from constants import SEG_DIR, METRICS_DIR, PAM50_METRICS_FILE, LABELED_SEG_DIR, LABELED_SEG_IMG, SEG_IMG, METRICS_FILE, IMG
from compute_torsion import compute_torsion_metric


def generate_segmentation(image_path, segmentation_folder='generated_data/seg'):
    """
    Generate a segmentation of a spinal cord image using SCT's deep segmentation tool.

    Args:
        image_path (str): Path to the input MRI image.
        segmentation_folder (str, optional): Folder to save the generated segmentation. Defaults to 'generated_data/seg'.
    """
    segmentation_cmd = ['sct_deepseg_sc', '-i', image_path, '-c', 't2', '-ofolder', segmentation_folder]
    subprocess.run(segmentation_cmd)


def create_labels(image_path, segmentation_path, labeled_segmentation_folder='generated_data/labeled_seg'):
    """
    Create vertebral labeling for an MRI image based on its segmentation.

    Args:
        image_path (str): Path to the input MRI image.
        segmentation_path (str): Path to the MRI image's segmentation.
        labeled_segmentation_folder (str, optional): Folder to save the labeled segmentation. Defaults to 'generated_data/labeled_seg'.
    """
    labeling_cmd = ['sct_label_vertebrae', '-i', image_path, '-s', segmentation_path, '-c', 't2', '-ofolder', labeled_segmentation_folder]
    subprocess.run(labeling_cmd)


def extract_metrics(segmentation_path: str, labeled_segmentation_path: str, output_metrics_path: str):
    """
    Process a spinal cord segmentation image and extract metrics. 
    sct_process_segmentation returns the metrics associated with segmentation in native space. 

    Args:
        segmentation_path (str): Path to the input spinal cord segmentation image.
        labeled_segmentation_path (str): Path to the labeled segmentation file.
        output_metrics_path (str): Path where the metrics will be saved.
    """

    get_metrics = ['sct_process_segmentation', '-i', segmentation_path, '-vertfile', labeled_segmentation_path, '-perslice', '1', '-o', output_metrics_path]
    subprocess.run(get_metrics)


def extract_pam50_metrics(segmentation_path: str, labeled_segmentation_path: str, output_pam50_path: str):
    """
    Process a spinal cord segmentation image and extract metrics.
    sct_process_segmentation returns the metrics associated with segmentation in PAM50 space. 

    Args:
        segmentation_path (str): Path to the input spinal cord segmentation image.
        labeled_segmentation_path (str): Path to the labeled segmentation image file.
        output_pam50_path (str): Path where the PAM50 metrics will be saved.
    """

    get_pam50_metrics = ['sct_process_segmentation', '-i', segmentation_path, '-vertfile', labeled_segmentation_path, '-perslice', '1', '-normalize-PAM50', '1', '-o', output_pam50_path]
    subprocess.run(get_pam50_metrics)


def clean_metrics_data(df):
    """
    Clean and reformat the PAM50 metrics data.

    Args:
        df (pd.DataFrame): The original DataFrame with PAM50 metrics data.

    Returns:
        pd.DataFrame: The cleaned and reformatted DataFrame.
    """
    print('Before cleaning:', df.iloc[0])  # Display the first row for reference

    # Define columns to drop and remove them
    columns_to_drop = ['SUM(length)', 'DistancePMJ', 'Filename', 'SCT Version', 'Timestamp']
    df_cleaned = df.drop(columns=columns_to_drop)

    # Remove columns that start with 'STD' and drop any rows with missing values
    df_cleaned = df_cleaned.filter(regex=r'^(?!STD)').dropna()

    # Rename columns: remove 'MEAN(' prefix and ')' suffix from column names
    df_cleaned = df_cleaned.rename(columns=lambda x: x.replace('MEAN(', '').replace(')', '') if 'MEAN(' in x else x)
    
    # Rename 'Slice (I->S)' column for clarity
    df_cleaned = df_cleaned.rename(columns={'Slice (I->S)': 'slice'})

    # Reset the DataFrame index for consistency
    df_cleaned = df_cleaned.reset_index(drop=True)

    print('After cleaning:', df_cleaned.iloc[0])  # Display the first row after cleaning

    return df_cleaned


def process_patient(patient):
    """
    Process an individual patient's data to generate and analyze spinal cord segmentations.

    Args:
        patient (str): Identifier of the patient.

    Returns:
        pd.DataFrame: Dataframe containing cleaned PAM50 metrics.
    """
    # Paths for the patient's files
    image_path = os.path.join(dataset_dir, patient, 'anat', patient + IMG)
    segmentation_path = os.path.join(SEG_DIR, patient + SEG_IMG)
    labeled_segmentation_path = os.path.join(LABELED_SEG_DIR, patient + LABELED_SEG_IMG)
    output_metrics_path = os.path.join(SEG_DIR, patient + METRICS_FILE)
    output_pam50_path = os.path.join(SEG_DIR, patient + PAM50_METRICS_FILE)

    # Segmentation and Labeling
    generate_segmentation(image_path)
    create_labels(image_path, segmentation_path)

    # Metrics Extraction
    extract_metrics(segmentation_path, labeled_segmentation_path, output_metrics_path)
    extract_pam50_metrics(segmentation_path, labeled_segmentation_path, output_pam50_path)

    # Error check for PAM50 metrics file
    if not os.path.exists(output_metrics_path):
        print('ERROR: Metrics file was not generated for', patient)
        return pd.DataFrame()

    return pd.read_csv(output_metrics_path)

if __name__ == '__main__':
    dataset_dir = os.getenv('DATASET_DIR')
    if not dataset_dir:
        raise ValueError("DATASET_DIR environment variable not set.")

    patients = [p for p in os.listdir(dataset_dir) if p.startswith('sub-')]

    # # Skip patients with known issues
    # patients = [p for p in patients if p not in ['sub-mgh01', 'sub-vallHebron04', 'sub-cmrra05']]

    for patient in patients:
        metrics_file_path = os.path.join(SEG_DIR, patient + METRICS_FILE)
        if os.path.exists(metrics_file_path):
            print('Skipping patient', patient)
            continue

        print('Processing patient', patient)
        metrics_df = process_patient(patient)
        metrics_df = clean_metrics_data(metrics_df)
        # Add torsion to metrics_df
        metrics_df = compute_torsion_metric(metrics_df)
        metrics_df.to_csv(os.path.join(METRICS_DIR, patient + '.csv'), index=False)