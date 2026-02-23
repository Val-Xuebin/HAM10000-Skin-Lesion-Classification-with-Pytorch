import os
import sys
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def load_image_paths(data_dir):
    """Load all image paths and create image_id to path mapping"""
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x 
        for x in all_image_path
    }
    return imageid_path_dict


def prepare_dataframe(metadata_path, imageid_path_dict, lesion_type_dict):
    """Load metadata and add path, cell_type, cell_type_idx columns"""
    df_original = pd.read_csv(metadata_path)
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    return df_original


def identify_duplicates(df_original):
    """Identify duplicated and unduplicated lesion_ids"""
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'
    
    df_original['duplicates'] = df_original['lesion_id']
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    return df_original


def split_train_val_test(df_original):
    """Split data into train (70%), validation (20%), and test (10%) sets"""
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    
    y = df_undup['cell_type_idx']
    # First split: 70% train, 30% temp (val + test)
    df_train, df_temp = train_test_split(
        df_undup, 
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    # Second split: split temp into 20% val and 10% test
    # Calculate the ratio for the second split
    y_temp = df_temp['cell_type_idx']
    val_ratio = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_ratio),  # This gives us test_size / (val_size + test_size)
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )
    
    # Map all data (including duplicates) to train/val/test
    def get_split(x):
        train_list = list(df_train['image_id'])
        val_list = list(df_val['image_id'])
        test_list = list(df_test['image_id'])
        if str(x) in train_list:
            return 'train'
        elif str(x) in val_list:
            return 'val'
        elif str(x) in test_list:
            return 'test'
        else:
            # Duplicates go to train set
            return 'train'
    
    df_original['split'] = df_original['image_id'].apply(get_split)
    
    df_train_all = df_original[df_original['split'] == 'train']
    df_val_all = df_original[df_original['split'] == 'val']
    df_test_all = df_original[df_original['split'] == 'test']
    
    return df_train_all, df_val_all, df_test_all


def balance_classes(df_train, data_aug_rate):
    """Balance classes by duplicating minority class samples"""
    for i in range(len(data_aug_rate)):
        if data_aug_rate[i]:
            class_data = df_train.loc[df_train['cell_type_idx'] == i, :]
            duplicated_data = [class_data] * (data_aug_rate[i] - 1)
            df_train = pd.concat([df_train] + duplicated_data, ignore_index=True)
    return df_train

