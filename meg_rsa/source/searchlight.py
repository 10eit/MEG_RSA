import os
import mne
import numpy as np
import utils.calculate_rdm
import utils.searchlight_tools
import utils.similarity
from joblib import Parallel, delayed
import numpy as np
import mne

def vertices_searchlight(stc, condition_label, model_rdm, src, radius, 
                         rdm_metric, rsa_metric, n_jobs=1):
    """
    Parallelized searchlight RSA analysis across cortical vertices.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate data (n_epochs, n_vertices, n_timepoints).
    condition_label : array-like
        Condition labels for each epoch (n_epochs,).
    model_rdm : numpy.ndarray
        Model RDM (n_conditions × n_conditions).
    src : mne.SourceSpaces
        Source space information.
    radius : float
        Searchlight radius in meters.
    rdm_metric : str
        Metric for neural RDM computation.
    rsa_metric : str
        Metric for RDM comparison.
    n_jobs : int
        Number of parallel jobs (default: 1).

    Returns
    -------
    numpy.ndarray
        RSA values for each vertex (n_vertices,).
    """
    stc_data = stc.data
    n_epochs, n_vertices, n_times = stc_data.shape

    # Split data by condition and average
    unique_conds = np.unique(condition_label)
    split_stc_data = [np.mean(stc_data[condition_label == cond], axis=0) 
                      for cond in unique_conds]

    # Pre-compute adjacency
    src_adjmat = mne.spatial_src_adjacency(src)

    # Inner function for parallel processing
    def _process_vertex(vertex_idx):
        patches = utils.searchlight_tools.source_patches(
            src=src_adjmat, k=vertex_idx, d=radius
        )
        if len(patches) == 0:
            return np.nan

        patch_data = [data[patches] for data in split_stc_data]
        neural_rdm = utils.calculate_rdm.construct_rdm(
            condition_list=patch_data,
            square_form=False,
            distance=rdm_metric
        )
        return utils.similarity.compute_similarity(
            method=rsa_metric,
            rdm1=model_rdm,
            rdm2=neural_rdm
        )

    # Parallel computation
    rsa_vertices = Parallel(n_jobs=n_jobs)(
        delayed(_process_vertex)(i) for i in range(n_vertices)
    )

    return np.array(rsa_vertices)

def roi_searchlight(data, model_rdm, src, fs_root, atlas, rdm_metric, rsa_metric):
    """
    data: morphed source estimate data, in shape of (n_trials,n_vertices,n_timepoints)
    model_rdm : compared model rdm, in shape of (n_trials,n_trials) or condensed form
    src : template source space you use
    fs_root : path to freesurfer directory
    atlas : parcellation you want
    rdm_metric : metric for computing rdm, accept augument support by scipy.spatial.distance.pdist() function
    rsa_metric : 
    """
    src_vertices = src[0]['vertno']  # Extract vertex indices from the source space
    subject_name = src.subject
    atlas_labels = mne.read_labels_from_annot(subject_name, parc=atlas, subjects_dir=fs_root, verbose='error')
    atlas_vertices = [label.vertices for label in atlas_labels]  # Extract vertices for each label
