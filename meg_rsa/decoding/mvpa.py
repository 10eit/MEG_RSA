from sensors import spatial
from sensors import spatial
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from joblib import Parallel, delayed


def sensor_searchlight_decoding(epochs, ch_type, y_label, clf, scorer, n_fold, average, radius, n_jobs=1):
    """
    Decoding on sensors with Spatial Smoothing using cross-validation and parallel processing.

    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object for decoding.
    ch_type : str
        Sensor type, 'grad' or 'mag' for Neuromag systems.
    y_label : None or numpy array
        Classification labels. If None, uses epochs.event_id.
    clf : sklearn classifier
        Classifier or pipeline implementing 'fit' and 'predict'.
    scorer : str or callable
        Scoring metric (e.g., 'accuracy' or scorer function).
    n_fold : int
        Number of cross-validation folds. Use -1 for Leave-One-Out.
    average : bool
        Return average score if True, else return all fold scores.
    radius : int
        Spatial smoothing radius (number of adjacent sensors).
    n_jobs : int
        Number of parallel jobs (default: 1).

    Returns:
    --------
    metric_overall : numpy array
        Decoding metrics for each sensor (averaged or per-fold).
    """

    # Extract data and validate inputs
    data_origin = epochs.get_data()  # Shape (n_epochs, n_channels, n_times)
    info = epochs.info
    n_epochs, n_channels, _ = data_origin.shape

    # Validate channel type and count for Neuromag
    if info['device_info']['type'] == 'TRIUX':
        check_params = {'mag': 102, 'grad': 204}
        assert n_channels == check_params[ch_type], \
            f"Expected {check_params[ch_type]} {ch_type} channels, got {n_channels}"

    # Prepare labels
    if y_label is None:
        y = epochs.events[:, 2]  # Use event IDs
    else:
        y = np.asarray(y_label)
        assert len(y) == n_epochs, "y_label length must match number of epochs"

    # Configure cross-validation
    if n_fold == -1:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    # Define per-sensor processing
    def process_sensor(i):
        try:
            # Find neighboring sensors
            neighbors = spatial.calculate_sensor_patches(info, ch_type, i, radius)
            if not neighbors.size:
                return np.nan  # No neighbors found

            # Extract and reshape data (n_epochs, [neighbors * times])
            X = data_origin[:, neighbors, :].reshape(n_epochs, -1)

            # Cross-validate
            scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer, n_jobs=1)
            return np.mean(scores) if average else scores

        except Exception as e:
            print(f"Sensor {i} failed: {str(e)}")
            return np.nan if average else [np.nan] * (n_epochs if n_fold == -1 else n_fold)

    # Parallel execution across sensors
    metrics = Parallel(n_jobs=n_jobs)(
        delayed(process_sensor)(i) for i in range(n_channels))
    
    return np.array(metrics)