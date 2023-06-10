
import os
import numpy as np

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from SSB.utils import load_config

config = load_config()
aircraft_root_dir = os.path.join(config['aircraft_directory'], 'fgvc-aircraft-2013b')

def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root=aircraft_root_dir, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.')

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)
    


def subsample_dataset(dataset: FGVCAircraft, 
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

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset: FGVCAircraft, 
                      include_classes):

    """
    Take a dataset, and keep only instances from selected classes

    Arguments:
    dataset --  dataset to subsample
    include_classes -- List or classes to keep

    Returns:
    dataset -- subsampled datdaset
    """

    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset


def get_train_val_indices(train_dataset: FGVCAircraft, 
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


    all_targets = [t for i, (p, t) in enumerate(train_dataset.samples)]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

