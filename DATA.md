# Contents

:red_car: [Supported datasets](#supported-datasets)

:ledger: [Config file](#config-file)

:arrow_down: [Download and pre-process](#download-and-pre-process)

:pencil: [Annotation format](#annotation-formats)

:file_folder: [Dataset directory formats](#dataset-directory-formats)

:clipboard: [Citation](#citations)

# Supported datasets

There are four datasets in the SSB: three fine-grained datasets ([CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Stanford Cars](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder), [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)); and [ImageNet](https://www.image-net.org/index.php).
Each dataset comes with pre-defined Known classes and Unknown classes, where Unknown classes are stratified into 'Easy' and 'Hard' based on their semantic similarity to the Known classes.

For the fine-grained datasets, the original classes are split into Known and Unknown subsets.
For ImageNet, the ImageNet-1K classes (ILSVRC12 challenge) are used as Known, and specific classes from [ImageNet-21K-P](https://arxiv.org/abs/2104.10972) are selected as Unknown.

**NOTE: For the ImageNet splits, you will need around 0.5TB of free disk space.** 

**NOTE: In some cases, a 'Medium' Unknown split is also specified. During benchmarking, these classes are combined into the 'Hard' split.** 

# Config file

A config file is expected in ```~/.ssb/ssb_config.json```. It specifies where each dataset should be saved to and read from.
If the datasets are already present [in the correct format](#dataset-directory-formats), you can point to them in this config.

There should be one entry for each of: CUB, Stanford Cars, FGVC-Aircraft, ImageNet-1K, ImageNet-21K.

By default, it is:

```
{
    "cub_directory": "~/data/CUB",
    "aircraft_directory": "~/data/FGVC_Aircraft",
    "scars_directory": "~/data/Stanford_Cars",
    "imagenet_1k_directory": "~/data/ImageNet-1K",
    "imagenet_21k_directory": "~/data/ImageNet-21K"
}
```

# Download and pre-process

To download and pre-process all datasets, you can run the following (note ImageNet-1K and ImageNet-21K must both be downlaoded explicitly):

```
>> from SSB.download import download_datasets
>> download_datasets(['cub', 'aircraft', 'scars', 'imagenet_1k', 'imagenet_21k'])
```

**Kaggle**: For some datasets, you will need a Kaggle account and API key set up (see README). 
For ImageNet, you will also need to join the competition.

**FGVC datasets**: These datasets are relative small and should download and pre-process in a few minutes. The datasets require only a few GB of disk space.

**ImageNet-1K**: This dataset takes a few hours to download and pre-process. It will need 200GB of disk space.

**ImageNet-21K**: This dataset is downlaoded in parallel for each synset. You can set the number of workers depending on your system bandwidth and specifications. You will need around 0.5TB of disk space.

# Annotation Formats:

For each dataset in (CUB, Stanford Cars, FGVC-Aircraft, ImageNet), there is an associated ```{dataset_name}_ssb_splits.json``` file in ```SSB/splits```.
When loaded, the dictionary has keys of: ```known_classes```, ```unknown_classes```, ```known_unknown_pairs```:

* ```known_classes``` is a list of all Known class indices
* ```unknown_classes``` is a itself a dictionary. It contains keys for 'Easy', 'Medium' (if applicable), and 'Hard', with lists of Unknown class indices.
* ```known_unknown_pairs``` (if applicable): For some datasets, for each Unknown class, we specify the Known class which is semantically most similar.

In ```SSB/splits```, we further include:
* ```index_to_class_name.json```. This maps each class index onto a readable class name.
* ```imagenet_21k_val_files.json```. For each synset in the Unknown ImageNet splits, this specifies the files which are reserved for validation.

## A note on class index formats:


* CUB: Class indices are given 0 - 199. These correspond to 1 - 200 in the original dataset.

* Stanford Cars: Classes ordered based on the order of ```cars_meta.mat``` in the original dataset. **Note that this ordering is different to the one recovered by running a standard PyTorch ImageFolder constructor in the root directory**.

* FGVC-Aircraft: Class indices are given as in the original dataset.

* ImageNet: Class indices are specified as WordNet synsets.

# Dataset directory formats

Utilities are included to download all datasets in the correct format automatically. 
Otherwise if you already have the dataset in the correct format, you can point to it in the config.

The formats are usually in the default style for the respective datasets and are detailed below.
The exception is ImageNet, which is reformated from the Kaggle version to be compatible with PyTorch ImageFolder.

## CUB

The directory under the root specified in the config should look like:

```
├── attributes.txt
└── CUB_200_2011
    ├── attributes
    ├── bounding_boxes.txt
    ├── classes.txt
    ├── image_class_labels.txt
    ├── images
    ├── images.txt
    ├── parts
    ├── README
    └── train_test_split.txt
```

## Stanford Cars

The directory under the root specified in the config should look like:

```
├── anno_test.csv
├── anno_train.csv
├── car_data
│   └── car_data
│       ├── test
│       └── train
└── names.csv
```

**Note**: You should not naively run a PyTorch ImageFolder constructor on the relevant data on the 'train' and 'test' directories.
The class indices this returns do not match those in the original dataset, and **do not match those in the SSB splits**.

## FGVC-Aircraft

The directory under the root specified in the config should look like:

```
└── fgvc-aircraft-2013b
    ├── data
    ├── evaluation.m
    ├── example_evaluation.m
    ├── README.html
    ├── README.md
    ├── vl_argparse.m
    ├── vl_pr.m
    ├── vl_roc.m
    └── vl_tpfp.m
```

## ImageNet-1K

The directory under the root specified in the config should look like:

```
├── train
|    └── nXXXXXXXX
|    └── nXXXXXXXX
|    └── ...
└── val
     └── nXXXXXXXX
     └── nXXXXXXXX
     └── ...
```

This will require refactoring if you have downloaded ImageNet through the Kaggle API (utilities are provided in SSB/download.py).

## ImageNet-21K

The directory under the root specified in the config should look like:

```
├── imagenet21k_train
|    └── nXXXXXXXX
|    └── nXXXXXXXX
|    └── ...
└── imagenet21k_val
     └── nXXXXXXXX
     └── nXXXXXXXX
     └── ...
```

We download specific synsets from ImageNet-21K.
We further reserve a specific set of images for a disjoint validation set (these should be the same validation images you would get by running the [ImageNet-21K-P](https://arxiv.org/abs/2104.10972) processing script).

# Citations

The citations for the original datasets (CUB, Stanford Cars, FGVC-Aircraft, ImageNet, ImageNet-21K-P) are:

```
@techreport{WahCUB_200_2011,
	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
	Author = {Catherine Wah and Steve Branson and Peter Welinder and Pietro Perona and Serge Belongie},
	Year = {2011},
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
```

```
@inproceedings{Cars196,
  title = {3D Object Representations for Fine-Grained Categorization},
  booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
  year = {2013},
  author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
}
```

```
@article{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {Subhransu Maji and Esa Rahtu and Juho Kannala and Matthew Blaschko and Andrea Vedaldi},
   year          = {2013},
  journal    = {arXiv preprint arXiv:1306.5151},
}
```

```
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
```

```
@misc{ridnik2021imagenet21k,
      title={ImageNet-21K Pretraining for the Masses}, 
      author={Tal Ridnik and Emanuel Ben-Baruch and Asaf Noy and Lihi Zelnik-Manor},
      year={2021},
      booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks}
}
```
