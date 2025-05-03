import os
import mne
import numpy as np
import meg_rsa.utils.rdm_tools
import utils.searchlight_tools
import meg_rsa.utils.similarity_tools
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
        neural_rdm = utils.rdm_tools.construct_rdm(
            condition_list=patch_data,
            square_form=False,
            distance=rdm_metric
        )
        return utils.similarity_tools.compute_similarity(
            method=rsa_metric,
            rdm1=model_rdm,
            rdm2=neural_rdm
        )

    # Parallel computation
    rsa_vertices = Parallel(n_jobs=n_jobs)(
        delayed(_process_vertex)(i) for i in range(n_vertices)
    )

    return np.array(rsa_vertices)

def roi_searchlight(data, model_rdm, src, fs_root, atlas, rdm_metric, rsa_metric, 
                   sanity_label=True, n_jobs=1):
    """
    ROI-based searchlight RSA analysis using anatomical atlases.

    Parameters
    ----------
    data : numpy.ndarray
        Source estimate data (n_trials, n_vertices, n_timepoints).
    model_rdm : numpy.ndarray
        Model RDM (n_conditions × n_conditions).
    src : mne.SourceSpaces
        Source space information.
    fs_root : str
        Path to FreeSurfer directory.
    atlas : str
        Parcellation/atlas name.
    rdm_metric : str
        Metric for neural RDM computation.
    rsa_metric : str
        Metric for RDM comparison.
    sanity_label : bool
        Whether to exclude labels with 'unknown' or '?' in name (default: True).
    n_jobs : int
        Number of parallel jobs (default: 1).

    Returns
    -------
    numpy.ndarray
        RSA values for each ROI (n_rois,).
    dict
        Label information for each ROI.
    """
    # Get source vertices and subject name
    src_vertices = src[0]['vertno']
    subject_name = src.subject
    
    # Read atlas labels
    atlas_labels = mne.read_labels_from_annot(
        subject_name, 
        parc=atlas, 
        subjects_dir=fs_root, 
        verbose='error'
    )
    
    # Filter labels if sanity check is enabled
    if sanity_label:
        atlas_labels = [
            label for label in atlas_labels 
            if 'unknown' not in label.name.lower() 
            and '?' not in label.name.lower()
        ]
    
    # Prepare ROI vertices (only those present in our source space)
    roi_vertices = []
    valid_labels = []
    
    for label in atlas_labels:
        # Intersect label vertices with our source space vertices
        roi_verts = np.intersect1d(label.vertices, src_vertices)
        if len(roi_verts) > 0:  # Only keep ROIs with vertices in our data
            roi_vertices.append(roi_verts)
            valid_labels.append(label)
    
    # Inner function for parallel processing
    def _process_roi(roi_verts):
        if len(roi_verts) == 0:
            return np.nan
        
        # Extract data for this ROI (average across vertices in ROI)
        roi_data = np.mean(data[:, roi_verts, :], axis=1)
        
        # Compute neural RDM
        neural_rdm = meg_rsa.utils.rdm_tools.construct_rdm(
            condition_list=roi_data,
            square_form=False,
            distance=rdm_metric
        )
        
        # Compare with model RDM
        return meg_rsa.utils.similarity_tools.compute_similarity(
            method=rsa_metric,
            rdm1=model_rdm,
            rdm2=neural_rdm
        )
    
    # Parallel computation across ROIs
    rsa_values = Parallel(n_jobs=n_jobs)(
        delayed(_process_roi)(roi_verts) for roi_verts in roi_vertices
    )
    
    return np.array(rsa_values), valid_labels

def spatiotemporal_searchlight(data)