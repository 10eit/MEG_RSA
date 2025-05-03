import os
import torch
import librosa
import transformers
from transformers import AutoModel, AutoProcessor

"""
If you want to calculate vectorized representation of your stimuli from a transfromer
It provides a helpful wrapper.
"""

def load_audio(stimulus_path,device='cpu'):
    """
    Loading Audio file and convert it to torch tensor with specific device
    
    Parameters
    ----------
    stimulus_path : PATH-like object | list
        PATH to the stimulus or a list of PATH to a dataset.
    device : str
        Tensor you want to load on, can be 'cuda' or 'cpu'.
        Default : 'cpu' to load tensor on CPU. 

    Returns
    -------
    float
        Pearson correlation coefficient between the two RDMs, ranging from -1 to 1.
    """
    if isinstance(stimulus_path,str):
        speech, _ = librosa.load(stimulus_path, sr=16000)
        return torch.from_numpy(speech).float().to(device)
    elif isinstance(stimulus_path,list):
        tensor_list = list()
        for path in stimulus_path:
            speech, _ = librosa.load(stimulus_path, sr=16000)
            tensor_list.append(torch.from_numpy(speech).float().to(device))
        return tensor_list

def fetch_phonetic_embedding(stimulus_path, device, model_name):
    audio_data = load_audio(stimulus_path,device=device)

    audio_processor = AutoProcessor.from_pretrained(model_name)
    audio_model = AutoModel.from_pretrained(model_name)

    if isinstance(stimulus_path,str):
        inputs = audio_processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = audio_model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze() 
    
    hidden_states = list()

    for data in audio_data:
        inputs = audio_processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = audio_model(**inputs)
        
        hidden_states.append(torch.mean(outputs.last_hidden_state, dim=1).squeeze())
    
    return hidden_states


def fetch_semantic_embedding(words, device, model_name):
    