from SSB.get_datasets.gcd_utils import MergedDataset
from SSB.get_datasets.stanford_cars import get_scars_gcd_datasets
from SSB.get_datasets.cub import get_cub_gcd_datasets
from SSB.get_datasets.fgvc_aircraft import get_aircraft_gcd_datasets
from SSB.get_datasets.imagenet import get_imagenet_easy_gcd_datasets, get_imagenet_hard_gcd_datasets

from SSB.utils import load_class_splits

from copy import deepcopy
import itertools

get_gcd_dataset_funcs = {
    'cub': get_cub_gcd_datasets,
    'aircraft': get_aircraft_gcd_datasets,
    'scars': get_scars_gcd_datasets,
    'imagenet_easy': get_imagenet_easy_gcd_datasets,
    'imagenet_hard': get_imagenet_hard_gcd_datasets
}


def get_gcd_datasets(dataset_name:str, train_transform, test_transform):

    """
    Return datasets for GCD

    Arguments:
        dataset_name: Name of dataset
        train_transform, test_transform: Callable PyTorch transforms for training and evaluation

    Returns:
        Tuple of datasets for training and evaluation
            train_dataset: Concatenated labelled and unlabelled subsets for training. D_U and D_L. Instance of MergedDataset
            test_dataset: Disjoint test set containing instances from Old and New classes
            unlabelled_train_examples_test: Only the unlabelled subset of the training set. D_U. Has the evaluation transform instead of train transform
            datasets: Dict of datasets originally returned by the get_{dataset_name}_gcd_dataset() functions
    """

    split_name = 'imagenet' if dataset_name in ('imagenet_easy', 'imagenet_hard') else dataset_name
    class_splits = load_class_splits(split_name)
    known_classes = class_splits['known_classes']
    
    if dataset_name in ('cub', 'aircraft', 'scars'):
        unknown_classes_ = [cls_ for diff, cls_ in class_splits['unknown_classes'].items()]
        unknown_classes = list(itertools.chain(*unknown_classes_))
    elif dataset_name == 'imagenet_easy':
        unknown_classes = class_splits['unknown_classes']['Easy']
    elif dataset_name == 'imagenet_hard':
        unknown_classes = class_splits['unknown_classes']['Hard']
    else:
        raise ValueError(f'dataset_name should be one of: (\'cub\', \'aircraft\', \'scars\', \'imagenet_easy\', \'imagenet_hard\'). \'{dataset_name}\' given. ')
        
    # Get datasets
    get_dataset_f = get_gcd_dataset_funcs[dataset_name]

    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            known_classes=known_classes,
                            prop_train_labels=0.5,
                            split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}

    if dataset_name in ('imagenet_easy', 'imagenet_hard'):

        # Datasets with samples from Imagenet21-K are designed to have labels only from indices 1000+
        n_classes_total = len(set(datasets['train_unlabelled'].targets))
        for i in range(n_classes_total):
            target_transform_dict[i] = i
    else:
        for i, cls in enumerate(known_classes + unknown_classes):
            target_transform_dict[cls] = i

    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets