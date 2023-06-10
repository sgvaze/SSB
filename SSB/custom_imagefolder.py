import torchvision
import numpy as np
from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS
from SSB.utils import load_index_to_name

class CustomImageFolder(torchvision.datasets.ImageFolder):

    """
    Base ImageFolder
    """

    def __init__(self, root, transform, dataset_name):

        ind_ = load_index_to_name()
        index_to_class_split = ind_[dataset_name]
        
        if dataset_name == 'scars':
            class_name_to_index = {name: int(ind) for ind, name in index_to_class_split.items()}
        else:
            raise ValueError

        samples = make_dataset(root, 
                               class_name_to_index, 
                               extensions=IMG_EXTENSIONS, 
                               is_valid_file=None)
        targets = [s[1] for s in samples]

        self.samples = samples
        self.targets = targets
        self.imgs = samples
        self.transform = transform
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


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