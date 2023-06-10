import os
import pandas as pd
import numpy as np

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from SSB.utils import load_config

config = load_config()
cub_root_dir = config['cub_directory']

class CustomCub2011(Dataset):

    """
    Base CUB-200-2011 dataset
    """

    base_folder = 'CUB_200_2011/images'

    def __init__(self, root=cub_root_dir, train=True, transform=None, target_transform=None, loader=default_loader):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)

        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]
    
def subsample_dataset(dataset: CustomCub2011, 
                      idxs):

    """
    Take a dataset, and keep only selected instances from it with in-place sub-sampling

    Arguments:
    dataset --  dataset to subsample
    idxs -- List or array of indices to subsample

    Returns:
    dataset -- subsampled datdaset
    """

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset

def subsample_classes(dataset: CustomCub2011, 
                      include_classes):

    """
    Take a dataset, and keep only instances from selected classes

    Arguments:
    dataset --  dataset to subsample
    include_classes -- List or classes to keep

    Returns:
    dataset -- subsampled datdaset
    """

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset

def get_train_val_indices(train_dataset: CustomCub2011, 
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

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs
