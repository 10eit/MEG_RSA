import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm  # Default tqdm
from tqdm.notebook import tqdm as tqdm_notebook  # Notebook-specific tqdm

def spatial_similarity(data, preselected_sensors=None, n_jobs=-1, use_notebook_tqdm=None):
    """
    Perform Spatial Representational Similarity Analysis (RSA).

    Parameters:
    - data: (n_trials, n_channels, n_timepoints) for single-subject or a list of such arrays for group-level.
    - preselected_sensors: List of selected sensor indices. Default is None (use all sensors).
    - n_jobs: Number of parallel jobs (default -1 for all cores).
    - use_notebook_tqdm: Whether to use tqdm.notebook. If None, automatically detects the environment.

    Returns:
    - r_values: Average off-diagonal R-values for data over timepoints.
    """
    def calculate_r_for_timepoint(t, data):
        """
        Helper function to compute RSA for a single timepoint.
        """
        # Extract data for the current timepoint
        spatial_vector = data[:, :, t]  # Shape: (n_trials, n_sensors)

        # Compute correlation matrices using np.corrcoef
        r_matrix = np.corrcoef(spatial_vector)  # Shape: (n_trials, n_trials)

        # Average off-diagonal values
        mask = ~np.eye(r_matrix.shape[0], dtype=bool)
        avg_r = np.mean(r_matrix[mask])

        return avg_r

    # Determine which tqdm to use
    if use_notebook_tqdm is None:
        try:
            # Check if running in a notebook environment
            get_ipython()
            use_notebook_tqdm = True
        except NameError:
            use_notebook_tqdm = False

    tqdm_func = tqdm_notebook if use_notebook_tqdm else tqdm

    # Handle group-level data (list of arrays)
    if isinstance(data, list):
        print("Calculating Group-level R-values")
        group_r_values = []

        # Iterate over each subject's data in the group
        for subject_data in data:
            if preselected_sensors is not None:
                subject_data = subject_data[:, preselected_sensors, :]

            n_timepoints = subject_data.shape[2]

            # Parallel computation across timepoints for this subject
            results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_r_for_timepoint)(t, subject_data) for t in tqdm_func(range(n_timepoints), desc='Spatial RSA')
            )

            group_r_values.append(np.array(results))

        # Average RSA values across subjects
        r_values = np.mean(group_r_values, axis=0)
        return r_values

    # Handle single-subject data (numpy array)
    elif isinstance(data, np.ndarray):
        if preselected_sensors is not None:
            data = data[:, preselected_sensors, :]

        n_timepoints = data.shape[2]

        # Parallel computation across timepoints
        results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_r_for_timepoint)(t, data) for t in tqdm_func(range(n_timepoints), desc='Spatial RSA')
        )

        r_values = np.array(results)
        return r_values

    else:
        raise TypeError("Input data must be a numpy array (single-subject) or a list of numpy arrays (group-level).")