import mne
import warnings
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

def source_patches(src, k, d=None, exclude_medial=None):
    """
    Generate a searchlight ball of vertices within distance d from a starting vertex k.

    Parameters
    ----------
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    k : int
        Starting vertex index.
    d : int, optional
        Distance (number of edges) to define the searchlight ball. If None, defaults
        are set based on ico-sampling (see below).
    exclude_medial : array-like, optional
        Array of vertex indices (e.g., medial wall vertices) to exclude from the ball.

    Returns
    -------
    searchlight_ball : np.ndarray
        Array of vertex indices within distance d from vertex k, excluding medial wall
        if specified.

    Notes
    -----
    If d is not provided, default distances are based on ico-sampling:
    - ico=7 (~0.7mm dipole spacing) -> d=9 (~6mm radius)
    - ico=6 (~1.5mm dipole spacing) -> d=4 (~6mm radius)
    - ico=5 (~3mm dipole spacing) -> d=2 (~6mm radius)
    - ico=4 (~6mm dipole spacing) -> d=1 (~6mm radius)
    - ico<4 -> d=1, with a warning suggesting univariate analysis.
    """
    # Check input types and get adjacency matrix
    if isinstance(src, mne.SourceSpaces):
        adjacency, _ = mne.channels.find_ch_adjacency(src, ch_type='meg')
    elif issparse(src):
        adjacency = src
    else:
        raise ValueError("src must be either an mne.SourceSpaces or a scipy.sparse.csr_matrix")
    
    # Ensure k is valid
    if not (0 <= k < adjacency.shape[0]):
        raise ValueError(f"Starting vertex k={k} is out of bounds for adjacency matrix of size {adjacency.shape[0]}")

    # Determine default distance d based on ico-sampling if not provided
    if d is None:
        if isinstance(src, mne.SourceSpaces):
            # Estimate ico-sampling based on number of vertices
            n_vertices = sum(len(h['vertno']) for h in src)
            if n_vertices > 100000:  # ico=7, ~0.7mm spacing
                d = 9  # ~6mm radius
            elif n_vertices > 40000:  # ico=6, ~1.5mm spacing
                d = 4
            elif n_vertices > 20000:   # ico=5, ~3mm spacing
                d = 2
            else:                     # ico=4 or lower, ~6mm spacing
                d = 1
                warnings.warn("Low ico-sampling detected (ico<4). Default d=1 is used. "
                              "Consider univariate analysis for better results.")
        else:
            d = 1  # Default for sparse matrix if no src info
            warnings.warn("No SourceSpaces provided, using default d=1. "
                          "Consider specifying d explicitly or using univariate analysis.")
    
    # Validate d
    if not isinstance(d, int) or d < 1:
        raise ValueError("Distance d must be a positive integer")

    # Initialize the starting node as a sparse row vector
    current = csr_matrix(([1], ([0], [k])), shape=(1, adjacency.shape[0]))
    
    # Keep track of all visited nodes
    visited_nodes = set(current.indices.tolist())
    
    # Perform matrix multiplication up to distance d
    for _ in range(d):
        current = current @ adjacency
        visited_nodes.update(current.indices.tolist())

    # Convert visited nodes to array
    searchlight_ball = np.array(list(visited_nodes))

    # Exclude medial wall vertices if specified
    if exclude_medial is not None:
        exclude_medial = np.asarray(exclude_medial)
        if not np.all(np.isin(exclude_medial, np.arange(adjacency.shape[0]))):
            raise ValueError("exclude_medial contains invalid vertex indices")
        searchlight_ball = np.setdiff1d(searchlight_ball, exclude_medial)
    
    return searchlight_ball

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