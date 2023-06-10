import numpy as np
from scipy import io as mat_io
import os

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from SSB.utils import load_config

config = load_config()
scars_root_dir = config['scars_directory']

car_root = os.path.join(scars_root_dir, 'cars_{}')
meta_default_path = os.path.join(scars_root_dir, 'devkit', 'cars_{}.mat')

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None, metas=meta_default_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):
        
        path = self.data[idx]

        image = self.loader(path)
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset: CarsDataset, 
                      idxs):

    """
    Take a dataset, and keep only selected instances from it with in-place sub-sampling

    Arguments:
    dataset --  dataset to subsample
    idxs -- List or array of indices to subsample

    Returns:
    dataset -- subsampled datdaset
    """

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset: CarsDataset, 
                      include_classes):

    """
    Take a dataset, and keep only instances from selected classes

    Arguments:
    dataset --  dataset to subsample
    include_classes -- List or classes to keep

    Returns:
    dataset -- subsampled datdaset
    """

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset

def get_train_val_indices(train_dataset: CarsDataset, 
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

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

