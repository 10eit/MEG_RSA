import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from scipy.sparse import csr_matrix, issparse

"""
This File intended for calculating spatial neighbour of single channel or vertices.
"""

def sensor_patches(info, ch_type=None, k=None, d=None):
    """
    Find the indices of nodes at distances less than or equal to d from node k 
    in a sparse adjacency matrix.

    Parameters:
    -----------
    info : mne.Info or scipy.sparse.csr_matrix
        Either an MNE Info object containing channel information, or a precomputed adjacency matrix.
    ch_type : str, optional
        The channel type to use for computing the adjacency matrix (e.g., 'grad' or 'mag').
        Required if info is an mne.Info object.
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
        If input types are invalid (info must be either mne.Info or csr_matrix).
        If ch_type is not provided when info is mne.Info.
    """
    # Check input types and get adjacency matrix
    if isinstance(info, mne.Info):
        if ch_type is None:
            raise ValueError("ch_type must be specified when info is an mne.Info object")
        adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type=ch_type)
    elif issparse(info) and info.format == 'csr':
        if ch_type is not None:
            import warnings
            warnings.warn("ch_type is ignored when info is a csr_matrix")
        adjacency = info
    else:
        raise ValueError("info must be either an mne.Info object or a scipy.sparse.csr_matrix")
    
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

def source_patches(src, k, d):
    """
    Find the indices of nodes at distances less than or equal to d from node k 
    in a sparse adjacency matrix.

    Parameters:
    -----------
    src : mne.SourceSpace or scipy.sparse.csr_matrix
        Either an MNE Source Space or a precomputed sparse adjacency matrix.
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
        If src is neither an mne.SourceSpace nor a scipy.sparse.csr_matrix.
    """
    # Check input types and get adjacency matrix
    if isinstance(src, mne.SourceSpace):
        adjacency, _ = mne.channels.find_src_adjacency(src)
    elif issparse(src) and src.format == 'csr':
        adjacency = src
    else:
        raise ValueError("src must be either an mne.SourceSpace or a scipy.sparse.csr_matrix")
    
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

def visualized_sensor_patches(
    info: Union[mne.Info, csr_matrix],
    ch_type: Optional[str] = None,
    k: Optional[int] = None,
    d: Optional[int] = None,
    title: Optional[str] = None,
    cmap: str = "Reds",
    highlight_color: str = "orange",
    figsize: tuple = (5, 5),
    show_names: bool = True
) -> plt.Figure:
    """
    Visualize spatial patches around a given node using a topographic map.

    Parameters:
    -----------
    info : mne.Info or scipy.sparse.csr_matrix
        Either an MNE Info object or a precomputed adjacency matrix.
    ch_type : str, optional
        The channel type (e.g., 'grad', 'mag', 'eeg'). Required if info is mne.Info.
    k : int, optional
        Index of the starting node. Required if info is mne.Info.
    d : int, optional
        Maximum distance to search for. Required if info is mne.Info.
    title : str, optional
        Custom title for the plot.
    cmap : str
        Colormap for the visualization.
    highlight_color : str
        Color for highlighting the patch channels.
    figsize : tuple
        Figure size in inches.
    show_names : bool
        Whether to show channel names (only shows starting channel if True).

    Returns:
    --------
    matplotlib.figure.Figure
        A figure showing the spatial patches on a topographic map.

    Raises:
    -------
    ValueError
        If required parameters are missing or invalid.
    """
    # Handle input types
    if isinstance(info, mne.Info):
        if None in (ch_type, k, d):
            raise ValueError("ch_type, k, and d must be provided when info is mne.Info")
        patches = sensor_patches(info, ch_type, k, d)
        ch_names = info['ch_names']
    elif isinstance(info, csr_matrix):
        if None in (k, d):
            raise ValueError("k and d must be provided when info is csr_matrix")
        patches = source_patches(info, k=k, d=d)
        ch_names = [f"Ch_{i}" for i in range(info.shape[0])]
    else:
        raise ValueError("info must be either mne.Info or scipy.sparse.csr_matrix")

    print(f"Starting channel: {ch_names[k]}")
    
    # Create name labeling function
    def name_label(x):
        """Label only the starting channel if show_names is True"""
        if show_names and x == ch_names[k]:
            return '★'  # Star symbol for starting channel
        return ""

    # Prepare data for visualization
    sim_data = np.full(info.shape[0] if isinstance(info, mne.Info) else info.shape[0], np.nan)
    mask = np.zeros_like(sim_data, dtype=bool)
    mask[patches] = True

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    
    # Plot topography
    if isinstance(info, mne.Info):
        evoked = mne.EvokedArray(sim_data[:, np.newaxis], info, tmin=0)
        evoked.plot_topomap(
            times=0,
            mask=mask[:, np.newaxis],
            axes=ax,
            cmap=cmap,
            vlim=(0, 1),  # Fixed range for better visualization
            show=False,
            colorbar=False,
            show_names=lambda x: name_label(x),
            mask_params=dict(
                markersize=10,
                markerfacecolor=highlight_color,
                markeredgecolor='k',
                markeredgewidth=0.5
            ),
            sensors=True,
            outlines='head'
        )
    else:
        # Fallback for CSR matrix (simplified plot)
        ax.scatter(
            np.arange(len(mask)),
            np.zeros(len(mask)),
            c=mask.astype(float),
            cmap=cmap,
            s=100,
            edgecolors='k'
        )
        ax.set_xticks(np.arange(len(mask)))
        ax.set_xticklabels(ch_names, rotation=90)
        ax.set_yticks([])
        ax.set_title("Adjacency Matrix Visualization" if not title else title)

    # Set title
    default_title = f'Spatial Patches (d={d})\n{len(patches)} Channels'
    ax.set_title(title if title else default_title, y=1.05)
    
    # Add legend
    ax.plot([], [], 'o', color=highlight_color, label='Patch channels', markersize=8)
    ax.plot([], [], 'k*', label='Starting channel', markersize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    return fig