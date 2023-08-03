import os
import json
import warnings
from pkg_resources import resource_filename
import numpy as np

DEFAULT_CONFIG = {
            "cub_directory": "~/data/CUB",
            "aircraft_directory": "~/data/FGVC_Aircraft",
            "scars_directory": "~/data/Stanford_Cars",
            "imagenet_1k_directory": "~/data/ImageNet-1K",
            "imagenet_21k_directory": "~/data/ImageNet-21K",
        }

def load_config() -> dict:

    """
    Loads SSB config.

    Returns: dict containing confid

    Details: 
    Config file specifies where datasets should be saved to / loaded from.
    E.g: 
        config = {
            "cub_directory": "~/data/CUB",
        }
    Expects a config file at ~/.ssb/ssb_config.json, otherwise it writes a default at this location 
    """

    # Choose an appropriate name and location for the config file
    json_root_dir = os.path.expanduser(os.path.join('~', '.ssb'))
    config_path = os.path.join(json_root_dir, 'ssb_config.json')

    if not os.path.exists(json_root_dir):
        os.makedirs(json_root_dir)

    if os.path.exists(config_path):
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config == DEFAULT_CONFIG:
            # Write default configuration to the file
            warnings.warn("The dataset path is set to the default value. "
                        "Consider setting it to a suitable path in your "
                        "~/.ssb/ssb_config.json file.", UserWarning)
    
    else:
        # Write default configuration to the file
        warnings.warn("The dataset path is set to the default value. "
                      "Consider setting it to a suitable path in your "
                      "~/.ssb/ssb_config.json file.", UserWarning)
        
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        config = DEFAULT_CONFIG

    return config

def load_class_splits(dataset_name: str) -> dict:
    """
    Load known/unknown class splits of a dataset.

    Arguments:
    dataset_name -- Which dataset to load.

    Returns: Dictionary with known classes and unknown classes
    """

    resource_relative_path = f'splits/{dataset_name}_ssb_splits.json'
    json_path = resource_filename('SSB', resource_relative_path)
    with open(json_path, 'r') as f:
        class_splits = json.load(f)

    return class_splits

def load_index_to_name() -> dict:

    """
    Loads dict of containing mapping between class index and name for each dataset
    """

    resource_relative_path = f'splits/index_to_class_name.json'
    json_path = resource_filename('SSB', resource_relative_path)
    with open(json_path, 'r') as f:
        class_splits = json.load(f)

    return class_splits

def subsample_instances(dataset, 
                        prop_indices_to_subsample=0.8):

    """
    Takes a dataset and samples a proportion of instances uniformly from it

    Arguments:
    dataset -- Should be a Pytorch Dataset object
    prop_indices_to_subsample -- What proportion of instances to sample (float between 0 and 1)

    Returns:
    subsample_indices -- numpy array of subsampled indices
    """

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def load_imagenet21k_val_files() -> dict:

    """
    Returns the files specified for validation from the ImageNet21-K-P classes
    Listed files are recovered if you follow instricutions from this paper: https://arxiv.org/abs/2104.10972
    """

    resource_relative_path = f'splits/imagenet_21k_val_files.json'
    json_path = resource_filename('SSB', resource_relative_path)
    with open(json_path, 'r') as f:
        class_splits = json.load(f)

    return class_splits