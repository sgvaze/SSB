import numpy as np
from copy import deepcopy
import os

from SSB.custom_imagefolder import CustomImageFolder, subsample_dataset, subsample_classes, get_train_val_indices, ConcatImageFolder
from SSB.utils import subsample_instances, load_config

config = load_config()
imagenet_1k_root = config['imagenet_1k_directory']
imagenet_21k_root = config['imagenet_21k_directory']

def get_imagenet_hard_gcd_datasets(known_classes, 
                     prop_train_labels=0.5,
                     train_transform=None, test_transform=None,  
                     split_train_val=False, 
                     seed=0):

    """
    Create PyTorch Datasets for ImageNet (Hard unknown classes from ImageNet-21K). 

    Arguments: 
    known_classes -- Unused, always the ImageNet 1K classes
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
    all_training_imgs_known_classes = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'train'),
            # root=os.path.join(imagenet_1k_root, 'val'),
            transform=train_transform,
            dataset_name='imagenet_1k'
        )
    all_training_imgs_unknown_classes = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_train'),
            # root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=train_transform,
            dataset_name='imagenet_21k_hard'
        )
    whole_training_set = ConcatImageFolder(dataset_1=all_training_imgs_known_classes, dataset_2=all_training_imgs_unknown_classes)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    known_classes_ = list(set(all_training_imgs_known_classes.targets))
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=known_classes_)
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

    # Init entire test set
    all_val_imgs_known_classes = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'val'),
            transform=test_transform,
            dataset_name='imagenet_1k'
        )
    all_val_imgs_unknown_classes = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=test_transform,
            dataset_name='imagenet_21k_hard'
        )
    test_dataset = ConcatImageFolder(dataset_1=all_val_imgs_known_classes, dataset_2=all_val_imgs_unknown_classes)

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



def get_imagenet_easy_gcd_datasets(known_classes, 
                     prop_train_labels=0.5,
                     train_transform=None, test_transform=None,  
                     split_train_val=False, 
                     seed=0):

    """
    Create PyTorch Datasets for ImageNet (Easy unknown classes from ImageNet-21K). 

    Arguments: 
    known_classes -- Unused, always the ImageNet 1K classes
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
    all_training_imgs_known_classes = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'train'),
            transform=train_transform,
            dataset_name='imagenet_1k'
        )
    all_training_imgs_unknown_classes = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_train'),
            transform=train_transform,
            dataset_name='imagenet_21k_easy'
        )
    whole_training_set = ConcatImageFolder(dataset_1=all_training_imgs_known_classes, dataset_2=all_training_imgs_unknown_classes)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    known_classes_ = list(set(all_training_imgs_known_classes.targets))
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=known_classes_)
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

    # Init entire test set
    all_val_imgs_known_classes = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'val'),
            transform=test_transform,
            dataset_name='imagenet_1k'
        )
    all_val_imgs_unknown_classes = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=test_transform,
            dataset_name='imagenet_21k_easy'
        )
    test_dataset = ConcatImageFolder(dataset_1=all_val_imgs_known_classes, dataset_2=all_val_imgs_unknown_classes)

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



def get_imagenet_osr_datasets(osr_split: str,
                              train_transform, test_transform,
                              eval_only=False, 
                              split_train_val=True):

    """
    Create PyTorch Datasets for ImageNet (Easy unknown classes from ImageNet-21K). 

    Loads datasets for open-set recognition

    Arguments: 
    osr_split -- Unused, always the ImageNet 1K classes
    train_transform, test_transform -- Torchvision transforms
    eval_only -- Only return test datasets from known / unknown classes
    split_train_val -- Unused flag
    
    Returns:
    all_datasets -- dict containing, 
        train_dataset: Training images from known classes
        val: Validation dataset from known classes (may be None)
        test_known: Test images from known classes
        test_unknown: Test images from unknown classes
    """

    np.random.seed(0)
    print('No validation split option for ImageNet dataset...')

    test_dataset_known = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'val'),
            transform=test_transform,
            dataset_name='imagenet_1k'
        )

    test_dataset_unknown = CustomImageFolder(
            root=os.path.join(imagenet_21k_root, 'imagenet21k_val'),
            transform=test_transform,
            dataset_name=f'imagenet_21k_{osr_split.lower()}'
        )

    if eval_only:

        train_dataset_known = None

    else:

        train_dataset_known = CustomImageFolder(
            root=os.path.join(imagenet_1k_root, 'train'),
            transform=train_transform,
            dataset_name='imagenet_1k'
        )

    all_datasets = {
        'train': train_dataset_known,
        'val': test_dataset_known,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets