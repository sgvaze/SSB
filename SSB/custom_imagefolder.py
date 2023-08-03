from typing import Any, Callable, Optional
import torchvision
import numpy as np
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
from torchvision.datasets.folder import find_classes as find_classes_default
from SSB.utils import load_index_to_name, load_class_splits

IMG_EXTENSIONS += ('.JPEG',)

class CustomImageFolder(torchvision.datasets.ImageFolder):

    """
    Base ImageFolder
    """

    def __init__(self, root, transform, dataset_name):

        self.root = root
        
        if dataset_name == 'scars':
            ind_ = load_index_to_name()
            index_to_class_split = ind_[dataset_name]
            class_name_to_index = {name: int(ind) for ind, name in index_to_class_split.items()}
        elif dataset_name == 'imagenet_21k_easy':
            class_splits = load_class_splits('imagenet')['unknown_classes']['Easy']
            class_name_to_index = {name: int(ind) + 1000 for ind, name in enumerate(class_splits)}      # Offset class indices by ImageNet1K clases
        elif dataset_name == 'imagenet_21k_hard':
            class_splits = load_class_splits('imagenet')['unknown_classes']['Hard']
            class_name_to_index = {name: int(ind) + 1000 for ind, name in enumerate(class_splits)}      # Offset class indices by ImageNet1K clases
        elif dataset_name == 'imagenet_1k':
            class_splits = load_class_splits('imagenet')['known_classes']
            class_name_to_index = {name: int(ind) for ind, name in enumerate(class_splits)}
        else:
            raise ValueError

        samples = make_dataset(root, 
                               class_name_to_index, 
                               extensions=IMG_EXTENSIONS, 
                               is_valid_file=None)

        self.imgs = samples
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = None            
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.class_to_idx = class_name_to_index

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


class ConcatImageFolder(CustomImageFolder):

    """
    Take two instances of CustomImageFolder and return an object
    with identical signtature to CustomImageFolder but with concatenated samples from both datasets
    """

    def __init__(self, dataset_1: CustomImageFolder, dataset_2: CustomImageFolder):

        samples_1, targets_1, cls_to_index_1 = dataset_1.samples, dataset_1.targets, dataset_1.class_to_idx
        samples_2, targets_2, cls_to_index_2 = dataset_2.samples, dataset_2.targets, dataset_2.class_to_idx

        # Combine samples, cls_to_index
        samples = samples_1 + samples_2
        class_name_to_index = {}
        for cls_1, t_1 in cls_to_index_1.items():
            class_name_to_index[cls_1] = t_1
        for cls_2, t_2 in cls_to_index_2.items():
            class_name_to_index[cls_2] = t_2

        # Assign attributes
        self.root = dataset_1.root                  # Assume both datasets have the same root
        self.imgs = samples
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = dataset_1.transform      # Assume both datasets have the same transform
        self.target_transform = None            
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.class_to_idx = class_name_to_index

        self.uq_idxs = np.array(range(len(self)))

def subsample_dataset(dataset: CustomImageFolder, 
                      idxs):

    """
    Take a dataset, and keep only selected instances from it with in-place sub-sampling

    Arguments:
    dataset --  dataset to subsample
    idxs -- List or array of indices to subsample

    Returns:
    dataset -- subsampled datdaset
    """

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset: CustomImageFolder, 
                      include_classes):

    """
    Take a dataset, and keep only instances from selected classes

    Arguments:
    dataset --  dataset to subsample
    include_classes -- List or classes to keep

    Returns:
    dataset -- subsampled datdaset
    """


    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset


def get_train_val_indices(train_dataset: CustomImageFolder, 
                          val_split=0.2):

    """
    Take a dataset, and sample indices for training and validation
    Each class is sampled proportionally

    Arguments:
    dataset --  dataset to subsample
    val_split -- Proportion of instances to reserve for validation

    Returns:
    train_idxs, val_idxs -- indices to reserve for training and validation
    """

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs