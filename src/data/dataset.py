import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


def download_kaggle_data(dataset_name='kmader/skin-cancer-mnist-ham10000', 
                         data_dir='input', 
                         force_download=False):
    """Download HAM10000 dataset from Kaggle"""
    data_path = Path(data_dir)
    metadata_path = data_path / 'HAM10000_metadata.csv'
    
    if not force_download and metadata_path.exists():
        img_dir1 = data_path / 'HAM10000_images_part_1'
        img_dir2 = data_path / 'HAM10000_images_part_2'
        if img_dir1.exists() and img_dir2.exists():
            print(f"Data already exists in {data_dir}, skipping download")
            return str(data_path)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset: {dataset_name}")
        print(f"Saving to: {data_path.absolute()}")
        
        data_path.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_name, path=str(data_path), unzip=True)
        
        print("Download completed")
        return str(data_path)
        
    except ImportError:
        print("Error: kaggle package not installed")
        print("Please run: pip install kaggle")
        raise
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        raise


def ensure_data_downloaded(data_dir='input', force_download=False):
    """Ensure data is downloaded, download if not exists"""
    data_path = Path(data_dir)
    metadata_path = data_path / 'HAM10000_metadata.csv'
    
    if not metadata_path.exists() or force_download:
        print("Data not found, attempting to download from Kaggle...")
        try:
            return download_kaggle_data(data_dir=data_dir, force_download=force_download)
        except Exception as e:
            print(f"Auto download failed: {e}")
            print("Please download data manually or check Kaggle API configuration")
            raise
    else:
        print(f"Data already exists in {data_dir}")
        return str(data_path)


class HAM10000(Dataset):
    """HAM10000 skin lesion classification dataset"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

