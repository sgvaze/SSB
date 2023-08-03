import os
import requests
import subprocess
import tarfile
import zipfile
import shutil
import json
from tqdm import tqdm
import concurrent.futures
import xml.etree.ElementTree as ET

from SSB.utils import load_config
from SSB.utils import load_class_splits
from SSB.utils import load_imagenet21k_val_files

CUB_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
AIRCRAFT_URL = 'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
CARS_COMMAND = 'kaggle datasets download -d jutrera/stanford-car-dataset-by-classes-folder'
IMAGENET_1K_COMMAND = 'kaggle competitions download -c imagenet-object-localization-challenge'

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

def download_and_unzip_imagenet_synset(wordnet_id, root_dir, val_files):
    
    # Step 1: Create directories
    train_dir = os.path.join(root_dir, 'imagenet21k_train', wordnet_id)
    val_dir = os.path.join(root_dir, 'imagenet21k_val', wordnet_id)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Check if all validation files are already present
    if all(os.path.exists(os.path.join(val_dir, filename)) for filename in val_files[wordnet_id]):
        return wordnet_id, []

    # Step 2: Download file
    url = f'https://image-net.org/data/winter21_whole/{wordnet_id}.tar'
    tar_file_path = os.path.join(train_dir, f'{wordnet_id}.tar')
    
    if not os.path.exists(tar_file_path):
        subprocess.run(['wget', '-O', tar_file_path, url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 3: Extract tar file
    with tarfile.open(tar_file_path, 'r') as tar_ref:
        tar_ref.extractall(path=train_dir)
    
    # Step 4: Delete tar file
    os.remove(tar_file_path)

    # Step 5: Move validation files
    missing_files = []
    for filename in val_files[wordnet_id]:
        src_path = os.path.join(train_dir, filename)
        dst_path = os.path.join(val_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            missing_files.append(filename)
    
    return wordnet_id, missing_files

def refactor_imagenet_1k(root_dir):

    """
    Refactors ImageNet1K dataset from format downloaded from Kaggle to one amenable for PyTorch ImageFolder
    At the end, structure should look like: '{root_dir}/train', '{root_dir}/val'
    Each of 'train' and 'val' directories should have sub-directories for each ImageNet-1K class containing correct images.
    """

    # Step 1: Move training images (already in the right format)
    old_train_dir = os.path.join(root_dir, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    new_train_dir = os.path.join(root_dir, 'train')
    shutil.move(old_train_dir, new_train_dir)

    # Step 2: Parse XML annotation files and move validation images
    # Originally, all validation files are in the same directory
    old_val_dir = os.path.join(root_dir, 'ILSVRC', 'Data', 'CLS-LOC', 'val')
    new_val_dir = os.path.join(root_dir, 'val')
    os.makedirs(new_val_dir, exist_ok=True)
    
    print('Refactoring ImageNet-1K validation files to correct structure...')
    annotations_dir = os.path.join(root_dir, 'ILSVRC', 'Annotations', 'CLS-LOC', 'val')
    for filename in tqdm(os.listdir(annotations_dir)):
        # Parse XML file
        tree = ET.parse(os.path.join(annotations_dir, filename))
        root = tree.getroot()
        wordnet_id = root.find("./object/name").text
        
        # Create new directory for this class if it doesn't exist
        class_dir = os.path.join(new_val_dir, wordnet_id)
        os.makedirs(class_dir, exist_ok=True)

        # Move image file
        img_filename = root.find("./filename").text + '.JPEG'
        old_img_path = os.path.join(old_val_dir, img_filename)
        new_img_path = os.path.join(class_dir, img_filename)
        shutil.move(old_img_path, new_img_path)

def download_and_unzip_imagenet21k(root_dir, max_workers=15):
    
    """
    Downloads all images from the 'Easy' and 'Hard' ImageNet unknown splits, from ImageNet21-K-P
    Creates training and validation subsets as obtained by running preprocessing from the ImageNet21-K-P paper: https://arxiv.org/abs/2104.10972
    Images are extracted into '{root_dir}/imagenet21k_train' and '{root_dir}/imagenet21k_val'
    Any files which are missing from the validation set are saved to ~/.ssb/missing_imagenet_21k_val_files.json

    root_dir: str
    max_workers: int How many synsets to download in parallel
        Change based on your operating system and bandwidth. Default 15 should be relatively stable.
    """

    # Get all classes in 'Easy' and 'Hard' unknown splits, from ImageNet21-K-P
    unknown_classes = load_class_splits('imagenet')['unknown_classes']
    unknown_classes = unknown_classes['Easy'] + unknown_classes['Hard']

    # Pre-specified files are reserved for validation from ImageNet-21K-P
    val_files = load_imagenet21k_val_files()

    # Download synsets in parallel.
    missing_files_all_synsets = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Initialize the progress bar
        pbar = tqdm(total=len(unknown_classes), desc="Downloading synsets", ncols=100)

        # Start the downloads
        future_to_synset = {executor.submit(download_and_unzip_imagenet_synset, synset, root_dir, val_files): synset for synset in unknown_classes}
        for future in concurrent.futures.as_completed(future_to_synset):

            synset = future_to_synset[future]
            try:
                synset, missing_files = future.result()
                missing_files_all_synsets[synset] = missing_files
            except Exception as exc:
                print('%r generated an exception: %s' % (synset, exc))
            
            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()
    
    # Save missing files
    json_root_dir = os.path.expanduser(os.path.join('~', '.ssb'))
    missing_file_json_path = os.path.join(json_root_dir, 'missing_imagenet_21k_val_files.json')

    # Now write the missing files to a json file
    with open(missing_file_json_path, 'w') as f:
        json.dump(missing_files_all_synsets, f, indent=4)

    print(f'Missing ImageNet21-K validation files written to {missing_file_json_path}')

def download_and_unzip_imagenet1k(directory):

    def _check_exists():
        return os.path.exists(os.path.join(directory, 'train')) and os.path.exists(os.path.join(directory, 'val'))

    if _check_exists():
        print('ImageNet-1K already downloaded')
        return

    print('Downloading ImageNet-1K...')
    command = IMAGENET_1K_COMMAND
    subprocess.run(command.split() + ['-p', directory], check=True)

    print('Extracting ImageNet-1K...')
    zipfile_path = os.path.join(directory, 'imagenet-object-localization-challenge.zip')
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

    print('Refactoring ImageNet-1K...')
    refactor_imagenet_1k(root_dir=directory)

    # Delete zip file
    os.remove(zipfile_path)

    print('Done!')

def download_datasets(datasets_to_download):

    config = load_config()

    download_funcs = {
        'cub': download_and_unzip_cub,
        'aircraft': download_and_unzip_aircraft,
        'scars': download_and_unzip_scars,
        'imagenet_1k': download_and_unzip_imagenet1k,
        'imagenet_21k': download_and_unzip_imagenet21k
    }

    for dataset_name in datasets_to_download:

        directory = config.get(f'{dataset_name}_directory', None)
        if not directory:
            print(f"Directory not specified for {dataset_name}. Skipping.")
            continue

        os.makedirs(directory, exist_ok=True)
        print(f'Downloading {dataset_name} dataset to: {directory}')
        download_funcs[dataset_name](directory)

        print(f"{dataset_name} downloaded and extracted successfully.")