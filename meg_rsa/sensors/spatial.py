### Importing Needed Packages
import mne
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def calculate_sensor_patches(info, ch_type, k, d):
    """
    Find the indices of nodes at distances less than or equal to d from node k 
    in a sparse adjacency matrix.

    Parameters:
    -----------
    info : mne.Info
        The MNE Info object containing channel information.
    ch_type : str
        The channel type to use for computing the adjacency matrix (e.g., 'grad' or 'mag').
    k : int
        Index of the starting node.
    d : int
        Maximum distance to search for.

    Returns:
    --------
    np.ndarray
        Indices of nodes at distances ≤ d from node k.

    Raises:
    -------
    ValueError
        If distance d is not a positive integer.
    """
    # Compute adjacency matrix based on channel type
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type=ch_type)

    if d < 1:
        raise ValueError("Distance d must be a positive integer")
    
    # Initialize the starting node as a sparse row vector
    current = csr_matrix(([1], ([0], [k])), shape=(1, adjacency.shape[0]))
    
    # Keep track of all visited nodes
    visited_nodes = set(current.indices.tolist())
    
    # Perform matrix multiplication up to distance d
    for _ in range(d):
        current = current @ adjacency
        visited_nodes.update(current.indices.tolist())
    
    return np.array(list(visited_nodes))

def check_spatial_patches(info, ch_type, k, d):
    """
    Visualize spatial patches around a given node using a topographic map.

    Parameters:
    -----------
    info : mne.Info
        The MNE Info object containing channel information.
    ch_type : str
        The channel type to use for computing the adjacency matrix (e.g., 'grad' or 'mag').
    k : int
        Index of the starting node.
    d : int
        Maximum distance to search for.

    Returns:
    --------
    matplotlib.figure.Figure
        A figure showing the spatial patches on a topographic map.
    """
    # Calculate spatial patches
    patches = calculate_sensor_patches(info=info, ch_type=ch_type, k=k, d=d)
    print(f"Starting channel: {info['ch_names'][k]}")
    
    def name_utils(x):
        """
        Helper function to label channels starting with the same prefix as the starting channel.
        """
        if x.startswith(info['ch_names'][k][:-1]):
            return 'HERE'
        else:
            return ""

    # Create simulated data and masks
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type=ch_type)
    sim_data = np.full(adjacency.shape[0], np.nan)
    mask = np.zeros((adjacency.shape[0], 1), dtype=bool)
    mask[patches, :] = True  # Mark all patches

    # Plot setup
    fig, ax_topo = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")
    evoked = mne.EvokedArray(sim_data[:, np.newaxis], info, tmin=0)
    
    # Plot base topomap
    evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=ax_topo,
        cmap="Reds",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        show_names=name_utils,
        mask_params=dict(markersize=10, markerfacecolor="orange", markeredgecolor="k"),
    )
    ax_topo.set_title(f'Spatial Patches \n {patches.shape[0]} Channels', y=0.9)
    return fig

def spatial_similarity(data, preselected_sensors=None, n_jobs=-1, use_notebook_tqdm=None):
    """
    Calculate Spatial Similarity within single condition.

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
    
def spatial_sensor_searchlight(info, ch_type, data, radius):
    """
    Computes spatial similarity across sensors using a searchlight approach.

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
        single_node_corr = spatial.spatial_similarity(
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