### Importing Needed Packages
import mne
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from adjacency_tools import calculate_sensor_patches

"""
Some helpful references for this tools:

1. https://elifesciences.org/articles/39061
2. https://www.jneurosci.org/content/40/16/3278
3. https://www.biorxiv.org/content/10.1101/2024.09.27.615440v1.full

This tool actually does not perform 'true RSA', 
it calculates spatial pattern consistency (similarity) in spatial pattern.

This tool can be used as temporal region selection to reduce multiple comparision
and computational cost. 
"""

def spatial_similarity(data, preselected_sensors=None, n_jobs=-1, use_notebook_tqdm=None):
    """
    Calculate Spatial (Pattern) Similarity within single condition.

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
        for subject_data in tqdm_func(data,desc='Processing Subjects'):
            if preselected_sensors is not None:
                subject_data = subject_data[:, preselected_sensors, :]

            n_timepoints = subject_data.shape[2]

            # Parallel computation across timepoints for this subject
            results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_r_for_timepoint)(t, subject_data) for t in range(n_timepoints)
            )

            group_r_values.append(np.array(results))

        # Average RSA values across subjects
        r_values = np.array(group_r_values)
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
    
def adjacency_similarity(info, ch_type, data, radius):
    """
    Computes spatial similarity across sensors and their adjacencies.

    Parameters:
    - info : mne.Info
        The MNE Info object containing channel information.
    - ch_type: str
        The channel type to use for computing the adjacency matrix (e.g., 'eeg' or 'meg').
    - data: 
        Single-subject: A NumPy array of shape (n_trials, n_channels, n_timepoints).
        Group-level: A list of such arrays for multiple subjects.
    - radius: int
        Radius (in sensor space) to define the neighborhood of each sensor.

    Returns:
    - corr_sensors: 
        Single-subject: A NumPy array of shape (n_channels,) containing the correlation values for each sensor.
        Group-level: A list of such arrays, one for each subject.
    """

    def within_patches_correlation(data, node, radius):
        """Calculate spatial similarity for a single node (sensor)."""
        neighbours = calculate_sensor_patches(info, ch_type, node, radius)
        single_node_corr = spatial_similarity(
            data, preselected_sensors=neighbours, n_jobs=-1, use_notebook_tqdm=None
        )
        return single_node_corr

    def compute_single_subject(data_subject):
        """Compute spatial similarity for a single subject."""
        _, channels, _ = data_subject.shape
        corr_sensors = np.zeros(channels)

        # Parallelize across channels for efficiency
        corr_sensors = Parallel(n_jobs=-1)(
            delayed(within_patches_correlation)(data_subject, i, radius)
            for i in range(channels)
        )

        return np.array(corr_sensors)

    # Check if the input data is a list (group-level)
    if isinstance(data, list):
        # Ensure all elements in the list have the correct shape
        if not all(isinstance(d, np.ndarray) and len(d.shape) == 3 for d in data):
            raise ValueError("All elements in the data list must be NumPy arrays of shape (n_trials, n_channels, n_timepoints).")

        # Compute results for each subject
        results = [compute_single_subject(subject_data) for subject_data in data]
        return results

    elif isinstance(data, np.ndarray):
        # Ensure the single-subject data has the correct shape
        if len(data.shape) != 3:
            raise ValueError("For single-subject data, the input array must have shape (n_trials, n_channels, n_timepoints).")

        # Compute and return the result for the single subject
        return compute_single_subject(data)

    else:
        raise TypeError("Input data must be a NumPy array or a list of NumPy arrays.")