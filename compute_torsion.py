# Credits to : Jan Valosek (https://github.com/spinalcordtoolbox/detect-compression/tree/main)

# If useful
def compute_torsion_metric(df):
    """Compute the torsion metric for a dataframe based on its orientation values."""
    
    # Create a new column 'torsion' initialized with NaN (more appropriate than empty string)
    df['torsion'] = None

    # Define a helper function to calculate torsion for a given index and range
    def torsion_for_range(idx, r):
        diffs = [abs(df['orientation'].iloc[idx] - df['orientation'].iloc[idx + i]) for i in r]
        return sum(diffs) / len(r)

    # Iterate over each index in the dataframe
    for idx in df.index:
        # List of indices for neighboring slices
        neighbor_indices = [-3, -2, -1, 1, 2, 3]

        # Filter out-of-bound indices and indices corresponding to None values
        valid_indices = [i for i in neighbor_indices if 0 <= idx + i < len(df) and df['orientation'].iloc[idx + i] is not None]
        
        if len(valid_indices) > 1:  # Ensure there's more than just the current slice
            df.at[idx, 'torsion'] = torsion_for_range(idx, valid_indices)

    return df