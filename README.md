# Install

For now:

```
git clone https://github.com/sgvaze/SSB.git
cd SSB
pip install -e .
```

# Set up JSON

Place config where data is currenlty location / should be downloaded in ```~/.ssb/ssb_config.json```

```
{
    "cub_directory": "/scratch/shared/beegfs/sagar/datasets/ssb/CUB", 
    "aircraft_directory": "/scratch/shared/beegfs/sagar/datasets/ssb/FGVC_Aircraft", 
    "scars_directory": "/scratch/shared/beegfs/sagar/datasets/ssb/Stanford_Cars/"
}
```

# Example commands

```
# Download FGVC datasets
>> from SSB.download import download_datasets
>> download_datasets(['cub', 'aircraft', 'scars'])

# Test GCD dataloaders
$ cd tests
$ python test_gcd_get_datasets.py

```

# Index constructions:

CUB-200-2011: class indices are given 0 - 199. Corresponding to 1 - 200 in the original dataset.

Stanford Cars: Classes ordered based on the order of ```cars_meta.mat``` in the original dataset.

# Kaggle

```
conda install -c conda-forge kaggle
```

You will need to pip install kaggle and log in to access the Stanford Cars dataset.

If you don't have an API key:
1. Go to Kaggle, create account and log in. 
2. Click top right icon, scroll to 'Settings'
3. Click 'Create New Token'

Place API key in home directory:
```
mkdir ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

# Other requirements

Formal requirements not included yet. But you will need:

```
torchvision
torch
scipy
```