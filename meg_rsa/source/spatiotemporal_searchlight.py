import os
import mne
import numpy as np

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


def run_single_searchlight(n_rdms, atlas, model_rdm, src, n_jobs):
    # Extract picklable data from src and atlas
    

    def process_single_rdm(i, src_vertices, atlas_vertices):
        # Generate a single RDM
        evoked = compute_sampled_evoked(i, group_labels, group_epochs)
        # Compare the RDM
        result = source_rdm_searchlight_roi(
            data=evoked, atlas_vertices=atlas_vertices, model_rdm=model_rdm, src_vertices=src_vertices
        )
        del evoked
        return result

    # Process each RDM in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_rdm)(i, src_vertices, atlas_vertices) for i in tqdm(range(n_rdms), desc="Running...", unit="iter")
    )

    # Convert results to a numpy array and apply Fisher Z transform
    result_array = np.array(results)
    z_array = fisher_z_transform(result_array).mean(axis=0)
    avg_r = inverse_fisher_z(z_array)

    return avg_r