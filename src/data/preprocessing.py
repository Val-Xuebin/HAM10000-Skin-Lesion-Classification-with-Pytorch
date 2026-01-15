import os
import sys
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

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


def split_train_val(df_original):
    """Split data into train and validation sets"""
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(
        df_undup, 
        test_size=config.VAL_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'
    
    df_original['train_or_val'] = df_original['image_id']
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    
    df_train = df_original[df_original['train_or_val'] == 'train']
    
    return df_train, df_val


def balance_classes(df_train, data_aug_rate):
    """Balance classes by duplicating minority class samples"""
    for i in range(len(data_aug_rate)):
        if data_aug_rate[i]:
            class_data = df_train.loc[df_train['cell_type_idx'] == i, :]
            duplicated_data = [class_data] * (data_aug_rate[i] - 1)
            df_train = pd.concat([df_train] + duplicated_data, ignore_index=True)
    return df_train

