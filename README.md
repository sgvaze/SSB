# Index constructions:

CUB-200-2011: class indices are given 0 - 199. Corresponding to 1 - 200 in the original dataset.
Stanford Cars: Classes ordered based on the order of ```cars_meta.mat``` in the original dataset.

# Other requirements

```
torchvision
torch
scipy
```

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