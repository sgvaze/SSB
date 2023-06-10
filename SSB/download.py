import os
import json
import requests
import subprocess
import tarfile
import zipfile

from SSB.utils import load_config
from SSB.utils import load_class_splits

CUB_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
AIRCRAFT_URL = 'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
CARS_COMMAND = 'kaggle datasets download -d jutrera/stanford-car-dataset-by-classes-folder'

def download_and_unzip_cub(directory, chunk_size=128):

    url = CUB_URL

    def _check_exists():
        return os.path.exists(os.path.join(directory, 'CUB_200_2011', 'images', '200.Common_Yellowthroat'))

    if _check_exists():
        print('CUB-200-2011 already downloaded')
        return

    print('Downloading CUB-200-2011...')
    save_path = os.path.join(directory, f"cub.tar.gz")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    
    print('Extracting CUB-200-2011...')
    with tarfile.open(save_path, 'r:gz') as tar:
        tar.extractall(path=directory)

def download_and_unzip_aircraft(directory, chunk_size=128):

    url = AIRCRAFT_URL

    def _check_exists():
        return os.path.exists(os.path.join(directory, 'fgvc-aircraft-2013b', 'data', 'images'))

    if _check_exists():
        print('FGVC-Aircraft already downloaded')
        return

    print('Downloading FGVC-Aircraft...')
    save_path = os.path.join(directory, f"aircraft.tar.gz")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    
    print('Extracting FGVC-Aircraft...')
    with tarfile.open(save_path, 'r:gz') as tar:
        tar.extractall(path=directory)

def download_and_unzip_scars(directory):

    def _check_exists():
        return os.path.exists(os.path.join(directory, 'cars_train', 'cars_train'))

    if _check_exists():
        print('Stanford Cars already downloaded')
        return

    print('Downloading Stanford Cars...')
    command = CARS_COMMAND
    subprocess.run(command.split() + ['-p', directory], check=True)

    print('Extracting Stanford Cars...')
    zipfile_path = os.path.join(directory, 'stanford-car-dataset-by-classes-folder.zip')
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

def download_datasets(datasets_to_download):

    config = load_config()

    download_funcs = {
        'cub': download_and_unzip_cub,
        'aircraft': download_and_unzip_aircraft,
        'scars': download_and_unzip_scars
    }

    for dataset_name in datasets_to_download:

        directory = config.get(f'{dataset_name}_directory', None)
        if not directory:
            print(f"Directory not specified for {dataset_name}. Skipping.")
            continue

        os.makedirs(directory, exist_ok=True)
        download_funcs[dataset_name](directory)

        print(f"{dataset_name} downloaded and extracted successfully.")