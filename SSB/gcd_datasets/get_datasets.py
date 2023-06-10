from SSB.gcd_datasets.gcd_utils import MergedDataset
from SSB.gcd_datasets.stanford_cars import get_scars_datasets
from SSB.gcd_datasets.cub import get_cub_datasets
from SSB.gcd_datasets.fgvc_aircraft import get_aircraft_datasets

from SSB.utils import load_class_splits

from copy import deepcopy
import itertools

get_dataset_funcs = {
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets
}


def get_datasets(dataset_name, train_transform, test_transform):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    class_splits = load_class_splits(dataset_name)
    known_classes = class_splits['known_classes']
    
    if dataset_name in ('cub', 'aircraft', 'scars'):
        unknown_classes_ = [cls_ for diff, cls_ in class_splits['unknown_classes'].items()]
        unknown_classes = list(itertools.chain(*unknown_classes_))
    else:
        raise ValueError(f'dataset_name should be one of: (\'cub\', \'aircraft\', \'scars\'). \'{dataset_name}\' given. ')
        

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]

    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=known_classes,
                            prop_train_labels=0.5,
                            split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
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