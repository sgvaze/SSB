import numpy as np
from copy import deepcopy

from SSB.custom_cub import CustomCub2011, subsample_dataset, subsample_classes, get_train_val_indices, cub_root_dir
from SSB.utils import subsample_instances

def get_cub_gcd_datasets(known_classes, 
                        prop_train_labels=0.5,
                        train_transform=None, test_transform=None,  
                        split_train_val=False, 
                        seed=0):

    """
    Create PyTorch Datasets for CUB. 

    Arguments: 
    known_classes -- List of integers, defines which classes are in the labelled set ('Old' classes)
    prop_train_labels -- What proportion of the 'Old' classes to include in the labelled dataset, D_L
    train_transform, test_transform -- Torchvision transforms
    split_train_val -- Whether to reserve some of the training set for validation. Returns None as validation instead 
    seed -- Numpy seed. Note: Set to zero for default implementation
    
    Returns:
    all_datasets -- dict containing, 
        labelled dataset with Old classes (D_L)
        unlabelled dataset with Old and New classes (D_U)
        validation set with Old and New classes
        test_dataset set with Old and New classes
    """

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=cub_root_dir, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=known_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCub2011(root=cub_root_dir, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

    