import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

def construct_rdm(condition_list, transformation=None, distance='euclidean', square_form=False):
    """
    Calculate Representation Dissimilarity Matrix (RDM) for a given condition list.

    Parameters:
    - condition_list: List or ndarray of condition parameters, shape (n_conditions, n_features).
    - transformation: callable, optional. Transformation applied to the condition_list.
    - distance: str, distance metric for pdist (default is 'euclidean').
    - square_form: bool, whether to return the matrix in square form (default is False).

    Returns:
    - rdm: ndarray, the computed RDM in either condensed or square form.
    """
    # Error handling
    if not isinstance(condition_list, (list, np.ndarray)):
        raise TypeError("condition_list must be a list or numpy array.")
    
    # Ensure condition_list is a 2D array
    condition_list = np.asarray(condition_list)
    if condition_list.ndim == 1:
        condition_list = condition_list.reshape(-1, 1)

    # Apply transformation if provided
    if transformation is not None:
        condition_list = transformation(condition_list)

    # Calculate pairwise distances
    rdm = pdist(condition_list, metric=distance)
    
    # Convert to square form if requested
    if square_form:
        rdm = squareform(rdm)
    
    return rdm


def plot_rdm(rdm, title="RDM", cmap="viridis", colorbar=True, labels=None):
    """
    Plot a Representation Dissimilarity Matrix (RDM).

    Parameters:
    - rdm: ndarray, the RDM to plot. Can be in either condensed or square form.
    - title: str, the title of the plot (default is "Representation Dissimilarity Matrix (RDM)").
    - cmap: str or matplotlib colormap, the colormap to use for the plot (default is "viridis").
    - colorbar: bool, whether to display a colorbar (default is True).
    - labels: list or array-like, labels for the conditions (default is None).

    Returns:
    - None (displays the plot).
    """
    # Error handling
    if not isinstance(rdm, np.ndarray):
        raise TypeError("rdm must be a numpy array.")
    
    if rdm.ndim != 1 and rdm.ndim != 2:
        raise ValueError("rdm must be a 1D (condensed) or 2D (square) array.")
    
    if rdm.ndim == 1:
        # Convert condensed form to square form
        try:
            rdm = squareform(rdm)
        except ValueError as e:
            raise ValueError("Invalid condensed distance matrix. Ensure the input is a valid condensed distance matrix.") from e
    
    if rdm.shape[0] != rdm.shape[1]:
        raise ValueError("Invalid square distance matrix. Ensure the input is a square matrix.")
    
    if labels is not None and len(labels) != rdm.shape[0]:
        raise ValueError("Length of labels must match the number of conditions in the RDM.")
    
    # Create the plot
    plt.figure(figsize=(3, 3))
    ax = plt.gca()  # Get the current axes
    rdm_img = ax.imshow(rdm, cmap=cmap, origin="upper")
    ax.set_title(title, fontsize=12)
    if colorbar:
        plt.colorbar(rdm_img, ax=ax, label="Dissimilarity")
        
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    return rdm_img