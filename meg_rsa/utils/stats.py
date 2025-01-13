import numpy as np
import mne
import meg_rsa.sensors.spatial as spatial


def compare_spatial_similarity(data_list, preselected_sensors=None, n_jobs=-1, use_notebook_tqdm=None):
    spatial_similarity_timecourse = [
        spatial.spatial_similarity(data, preselected_sensors=preselected_sensors, n_jobs=n_jobs, use_notebook_tqdm=use_notebook_tqdm) for data in data_list
    ]
    


