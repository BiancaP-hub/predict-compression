import subprocess
import os
from cord_sense.common.constants import COMPRESSION_DIR, SEG_DIR, LABELED_SEG_DIR, LABELED_SEG_IMG, SEG_IMG

def compute_mscc_from_compression_labels(segmentation_path, labeled_segmentation_path, compression_labels_path, output_mscc_path):
    """
    Compute the normalized Mean Spinal Canal Compression (MSCC) for given image paths.

    Parameters:
    - segmentation_path (str): Path to the segmentation image.
    - labeled_segmentation_path (str): Path to the labeled segmentation image.
    - compression_labels_path (str): Path to the compression labels.
    - output_mscc_path (str): Path for the output MSCC file.
    """
    command = [
        'sct_compute_compression', '-i', segmentation_path, 
        '-vertfile', labeled_segmentation_path, '-l', compression_labels_path, 
        '-metric', 'diameter_AP', '-normalize-hc', '1', '-o', output_mscc_path
    ]
    subprocess.run(command)


if __name__ == '__main__':
    for file in os.listdir(COMPRESSION_DIR):
        if file.endswith('.nii.gz'):
            patient_id = file.split('_')[0]
            seg_file = os.path.join(SEG_DIR, patient_id + SEG_IMG)
            labeled_seg_file = os.path.join(LABELED_SEG_DIR, patient_id + LABELED_SEG_IMG)
            output_path = 'generated_data/mscc_man.csv'  # Define the output path for MSCC results

            compute_mscc_from_compression_labels(
                os.path.join(COMPRESSION_DIR, file), 
                seg_file, 
                labeled_seg_file, 
                output_path
            )