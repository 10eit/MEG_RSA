import mne
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

def calculate_spatial_patches(info, ch_type, k, d):
    """
    Find the indices of nodes at distances less than or equal to d from node k 
    in a sparse adjacency matrix.

    Parameters:
    -----------
    info : mne.Info
        The MNE Info object containing channel information.
    ch_type : str
        The channel type to use for computing the adjacency matrix (e.g., 'eeg' or 'meg').
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
        The channel type to use for computing the adjacency matrix (e.g., 'eeg' or 'meg').
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
    patches = calculate_spatial_patches(info=info, ch_type=ch_type, k=k, d=d)
    print(f"Starting channel: {info['ch_names'][k]}")
    
    def name_utils(x):
        """
        Helper function to label channels starting with the same prefix as the starting channel.
        """
        if x.startswith(info['ch_names'][k][:-1]):
            return 'start'
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