from SSB.gcd_datasets.cub import get_cub_datasets
from SSB.gcd_datasets.fgvc_aircraft import get_aircraft_datasets
from SSB.gcd_datasets.stanford_cars import get_scars_datasets
from SSB.utils import load_class_splits

def test_cub():

    print('Testing CUB dataset construction')
    class_splits = load_class_splits('cub')
    known_classes = class_splits['known_classes']
    x = get_cub_datasets(known_classes=known_classes)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

def test_aircraft():

    print('Testing FGVC-Aircraft dataset construction')
    class_splits = load_class_splits('aircraft')
    known_classes = class_splits['known_classes']

    x = get_aircraft_datasets(known_classes=known_classes)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set([i[1] for i in x['train_labelled'].samples])))
    print('Printing total number of classes...')
    print(len(set([i[1] for i in x['train_unlabelled'].samples])))

def test_cars():

    print('Testing Stanford Cars dataset construction')
    class_splits = load_class_splits('scars')
    known_classes = class_splits['known_classes']

    x = get_scars_datasets(known_classes=known_classes)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

if __name__ == '__main__':

    test_cars()