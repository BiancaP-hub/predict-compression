def extract_patient_id(filename):
    # Split by '/' and then by '_', and take the second and first elements respectively
    return filename.split('/')[1].split('_')[0]