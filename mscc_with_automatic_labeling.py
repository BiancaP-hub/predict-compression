import json
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from constants import PARTICIPANTS_FILE, LABELED_SEG_DIR, LABELED_SEG_IMG

def apply_erosion(data, vertebra_label, structure=None):
    """
    Apply binary erosion to the data for a specific vertebral label.

    Parameters:
    - data (ndarray): The image data.
    - vertebra_label (int): The label of the vertebra.
    - structure (ndarray, optional): The structure for the erosion operation.

    Returns:
    - ndarray: Eroded image data for the specified vertebral label.
    """
    vertebra_data = (data == vertebra_label)
    eroded_data = binary_erosion(vertebra_data, structure=structure)
    return eroded_data


def get_most_compressed_slice_post_erosion(data, vertebra_label, structure=None):
    """
    Find the slice with the minimum area after erosion within the range of the labeled vertebra.

    Parameters:
    - data (ndarray): The image data.
    - vertebra_label (int): The label of the vertebra.
    - structure (ndarray, optional): The structure for the erosion operation.

    Returns:
    - int or None: The index of the most compressed slice or None if no slice is found.
    """
    eroded_data = apply_erosion(data, vertebra_label, structure)
    slice_range = get_slice_range_for_labeled_vertebrae(data, vertebra_label)

    if not slice_range or slice_range == (None, None):
        return None

    start_slice, end_slice = slice_range
    areas = [np.sum(eroded_data[:, :, i]) for i in range(start_slice, end_slice + 1)]

    min_area_index = areas.index(min(areas)) + start_slice
    return min_area_index


def get_slice_thickness(path_to_image):
    """
    Retrieve the slice thickness from the image header.

    Parameters:
    - path_to_image (str): Path to the image file.

    Returns:
    - float: The slice thickness.
    """
    img = nib.load(path_to_image)
    affine_matrix = img.affine
    return abs(affine_matrix[2, 2])


def get_slice_range_for_labeled_vertebrae(data, vertebral_label):
    """
    Retrieve the slice range for the specified vertebral label.

    Parameters:
    - data (ndarray): The image data.
    - vertebral_label (int): The vertebral label.

    Returns:
    - tuple: A tuple of (start_slice, end_slice) indices, or (None, None) if not found.
    """
    slices_with_label = np.any(data == vertebral_label, axis=(0, 1))
    slice_indices = np.where(slices_with_label)[0]

    return (slice_indices[0], slice_indices[-1]) if slice_indices.size > 0 else (None, None)


def get_healthy_region(slice_index, direction, sums, slice_thickness, distance=10, length=20):
    """
    Find the healthy region starting from a distance and spanning a certain length.

    Parameters:
    - slice_index (int): The index of the slice.
    - direction (int): The direction of analysis (-1 for downwards, 1 for upwards).
    - sums (list): Sum of labeled pixels for each axial slice.
    - slice_thickness (float): The thickness of each slice.
    - distance (float): The distance from the slice index to start analysis.
    - length (float): The length of the region to analyze.

    Returns:
    - float or None: The mean value of the healthy region or None if out of bounds.
    """
    slices_to_skip = int(distance / slice_thickness)
    slices_to_consider = int(length / slice_thickness)

    if direction == -1:  # moving downwards
        end_index = slice_index - slices_to_skip
        start_index = end_index - slices_to_consider
    else:  # moving upwards
        start_index = slice_index + slices_to_skip
        end_index = start_index + slices_to_consider

    start_index, end_index = max(0, start_index), min(len(sums) - 1, end_index)
    return None if end_index < start_index else np.mean(sums[start_index:end_index + 1])


def compute_mscc(path_to_segmented_image, compressed_vertebrae_range=None):
    """
    Compute the Mean Spinal Canal Compression (MSCC) values for a given image.

    Parameters:
    - path_to_segmented_image (str): Path to the segmented image file.
    - compressed_vertebrae_range (list): Range of vertebral levels to analyze.

    Returns:
    - dict: MSCC values for each compression site.
    """
    img = nib.load(path_to_segmented_image)
    data = img.get_fdata()
    slice_thickness = get_slice_thickness(path_to_segmented_image)

    if not slice_thickness:
        print("Invalid slice thickness encountered. Returning an empty dictionary.")
        return {}

    sums = np.sum(np.sum(data, axis=0), axis=0)  # sum of labeled pixels for each axial slice
    mscc_values = {}
    compressed_slices, compressed_slices_mapping = [], {}

    for vertebra in compressed_vertebrae_range:
        most_compressed_slice = get_most_compressed_slice_post_erosion(data, vertebra)
        if most_compressed_slice is not None:
            compressed_slices.append(most_compressed_slice)
            compressed_slices_mapping[most_compressed_slice] = vertebra

    if not compressed_slices:
        return {}

    highest_compressed_slice = max(compressed_slices)
    lowest_compressed_slice = min(compressed_slices)
    upper_healthy_mean = get_healthy_region(highest_compressed_slice, 1, sums, slice_thickness)
    lower_healthy_mean = get_healthy_region(lowest_compressed_slice, -1, sums, slice_thickness)

    for slice_idx in compressed_slices:
        vertebra = compressed_slices_mapping[slice_idx]
        diameter_compression = sums[slice_idx]
        mean_diameter_healthy = compute_mean_diameter_healthy(upper_healthy_mean, lower_healthy_mean, diameter_compression)
        mscc_value = max(0, 1 - (diameter_compression / mean_diameter_healthy))
        mscc_values[vertebra] = mscc_value

    return mscc_values

def compute_mean_diameter_healthy(upper_healthy_mean, lower_healthy_mean, diameter_compression):
    """
    Compute the mean diameter of the healthy region.

    Parameters:
    - upper_healthy_mean (float): Mean diameter of the upper healthy region.
    - lower_healthy_mean (float): Mean diameter of the lower healthy region.
    - diameter_compression (float): Diameter at the compression site.

    Returns:
    - float: The mean diameter of the healthy region.
    """
    if upper_healthy_mean < diameter_compression and lower_healthy_mean < diameter_compression:
        return None
    elif upper_healthy_mean < diameter_compression:
        return 2 * lower_healthy_mean
    elif lower_healthy_mean < diameter_compression:
        return 2 * upper_healthy_mean
    else:
        return (upper_healthy_mean + lower_healthy_mean) / 2


def compute_mscc_for_row(row):
    """
    Compute the MSCC for a given row from the participant data.

    Parameters:
    - row (DataFrame row): A row from the participant data.

    Returns:
    - float or dict: MSCC value or a dictionary of MSCC values, 0 if 'HC' pathology.
    """
    if row['pathology'] == 'HC':
        return 0

    patient = row['participant_id']
    labeled_segmentation_path = os.path.join(LABELED_SEG_DIR, patient + LABELED_SEG_IMG)
    match = re.search(r"C(\d+)[-/](\d+)", row['notes'])

    if match:
        start_label, end_label = map(int, match.groups())
        compressed_vertebrae_range = list(range(start_label, end_label + 1))
    else:
        return {}

    return compute_mscc(labeled_segmentation_path, compressed_vertebrae_range)


if __name__ == '__main__':
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    mscc_dict = {row['participant_id']: compute_mscc_for_row(row) for _, row in participants.iterrows()}

    with open('generated_data/mscc_auto.json', 'w') as fp:
        json.dump(mscc_dict, fp, indent=4)