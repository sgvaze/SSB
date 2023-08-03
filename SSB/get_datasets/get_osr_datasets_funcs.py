from SSB.get_datasets.imagenet import get_imagenet_osr_datasets

get_gcd_dataset_funcs = {
    'imagenet': get_imagenet_osr_datasets
}


def get_osr_datasets(dataset_name:str,
                     osr_split: str,
                     train_transform, test_transform,
                     eval_only: bool = False,
                     split_train_val: bool = True,
                     ):

    """
    Return datasets for Open Set Recognition.
    Only implemented for ImageNet currently.

    dataset_name: Must be 'imagenet'
    osr_split: Open-set split difficulty. Must be either 'Easy' or 'Hard'
    train_transform, test_transform: PyTorch image transforms
    eval_only: Set to true if only running evaluation (if the closed-training data is not required). Speeds up data loading.
    split_train_val: All datasets lack annotated train and val splits. Set to true to split the annotated validation data into 'val' and 'test' subsets.

    Returns a dict of PyTorch datasets:
        all_datasets = {
            'train': Closed-set training data (None if eval_only is True),
            'val': 'Closed-test validation data (None if split_train_val is False),
            'test_known': Closed-set test data,
            'test_unknown': Open-set test data,
        }
    """
    
    if dataset_name not in ('imagenet',):
        raise ValueError(f'dataset_name should be one of: (\'imagenet\',). \'{dataset_name}\' given. ')
        

    # Get datasets
    get_dataset_f = get_gcd_dataset_funcs[dataset_name]

    datasets = get_dataset_f(osr_split=osr_split,
                              train_transform=train_transform, test_transform=test_transform,
                              eval_only=eval_only, 
                              split_train_val=split_train_val)

    return datasets