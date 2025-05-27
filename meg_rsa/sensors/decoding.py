import utils
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier
from joblib import Parallel, delayed
import utils.searchlight_tools

def decode_vertex(idx, src, data, label, clf, cv, scoring_function, d=None, exclude_medial=None):
    """
    Perform decoding for a single vertex using its searchlight neighbors.

    Parameters
    ----------
    idx : int
        Index of the vertex to decode.
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    data : np.ndarray
        Source space data with shape (n_trials, n_vertices, n_timepoints).
    label : np.ndarray
        Labels for each trial with shape (n_trials,).
    clf : sklearn.base.BaseEstimator
        Scikit-learn classifier object (e.g., LogisticRegression).
    cv : sklearn.model_selection._BaseCrossValidator
        Cross-validation strategy.
    scoring_function : str or callable
        Scoring function for cross-validation (e.g., 'roc_auc', 'accuracy', or custom scorer).
    d : int, optional
        Distance for searchlight ball. If None, defaults are set in source_patches.
    exclude_medial : array-like, optional
        Vertex indices to exclude (e.g., medial wall vertices).

    Returns
    -------
    score : float
        Mean cross-validated decoding score for the vertex, or np.nan if decoding fails.
    """
    # Get neighboring vertices using source_patches
    neighbour_vertices = utils.searchlight_tools.source_patches(src, k=idx, d=d, exclude_medial=exclude_medial)
    
    # Ensure neighbor vertices are valid
    neighbour_vertices = neighbour_vertices[np.isin(neighbour_vertices, np.arange(data.shape[1]))]
    if len(neighbour_vertices) == 0:
        return np.nan
    
    # Extract data for the searchlight ball
    X = data[:, neighbour_vertices, :]  # Shape: (n_trials, n_neighbors, n_timepoints)
    
    # Flatten spatio-temporal data into a single feature vector per trial
    X_flat = X.reshape(data.shape[0], -1)  # Shape: (n_trials, n_neighbors * n_timepoints)
    
    # Compute cross-validated decoding score
    try:
        score = cross_val_score(clf, X_flat, label, cv=cv, scoring=scoring_function, n_jobs=1)
        return np.mean(score)  # Average across folds
    except ValueError as e:
        print(f"Warning: Decoding failed at vertex {idx}: {e}")
        return np.nan

def searchlight_decoding(src, data, label, clf, scoring_function='roc_auc', d=None, exclude_medial=None, cv=None, n_jobs=1):
    """
    Perform searchlight decoding on source space data, using the entire (n_vertices, n_timepoints)
    as a single feature vector for each trial, with parallel processing over vertices.

    Parameters
    ----------
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space object or adjacency matrix defining connectivity.
    data : np.ndarray
        Source space data with shape (n_trials, n_vertices, n_timepoints).
    label : np.ndarray
        Labels for each trial with shape (n_trials,).
    clf : sklearn.base.BaseEstimator
        Scikit-learn classifier object (e.g., LogisticRegression).
    scoring_function : str or callable, optional
        Scoring function for cross-validation (e.g., 'roc_auc', 'accuracy', or custom scorer).
        Defaults to 'roc_auc'.
    d : int, optional
        Distance for searchlight ball. If None, defaults are set in source_patches.
    exclude_medial : array-like, optional
        Vertex indices to exclude (e.g., medial wall vertices).
    cv : sklearn.model_selection._BaseCrossValidator, optional
        Cross-validation strategy. If None, defaults to 5-fold StratifiedKFold.
    n_jobs : int, optional
        Number of parallel jobs for vertex processing. Defaults to 1 (no parallelization).

    Returns
    -------
    scores : np.ndarray
        Decoding scores for each vertex, shape (n_vertices,).
    """
    # Input validation
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data must be a 3D numpy array (n_trials, n_vertices, n_timepoints)")
    if not isinstance(label, np.ndarray) or label.shape[0] != data.shape[0]:
        raise ValueError("label must be a 1D array with length equal to n_trials")
    if not is_classifier(clf):
        raise ValueError("clf must be a scikit-learn classifier")
    if not (isinstance(scoring_function, str) or callable(scoring_function)):
        raise ValueError("scoring_function must be a string or callable")
    if not isinstance(n_jobs, int) or n_jobs < -1 or n_jobs == 0:
        raise ValueError("n_jobs must be a positive integer, -1, or None")

    _, n_vertices, _ = data.shape

    # Default cross-validation strategy
    if cv is None:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define decoding vertices, excluding medial wall if specified
    decoding_vertices = np.arange(n_vertices)
    if exclude_medial is not None:
        exclude_medial = np.asarray(exclude_medial)
        if not np.all(np.isin(exclude_medial, decoding_vertices)):
            raise ValueError("exclude_medial contains invalid vertex indices")
        decoding_vertices = np.setdiff1d(decoding_vertices, exclude_medial)

    # Parallelize decoding across vertices
    scores = Parallel(n_jobs=n_jobs)(
        delayed(decode_vertex)(idx, src, data, label, clf, cv, scoring_function, d, exclude_medial)
        for idx in decoding_vertices
    )

    # Convert results to numpy array
    scores = np.array(scores)

    # Ensure scores array has correct shape (n_vertices,)
    scores_full = np.full(n_vertices, np.nan)
    scores_full[decoding_vertices] = scores

    return scores_full