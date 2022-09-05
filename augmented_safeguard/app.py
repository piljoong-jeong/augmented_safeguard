"""
### app.py

maintain entrypoints
"""

import os
import sys

import numpy as np
import pandas as pd
import seaborn

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import augmented_safeguard as asfgd
from augmented_safeguard.dataset import DatasetType, DatasetManager


def run_default():
    """
    ### run_default


    """

    dir_dataset = "/mnt/d/"
    type_dataset = DatasetType.RIO10
    name_train_sequence = "seq01_01"
    name_test_sequence = "seq01_02"

    dataset_manager = DatasetManager(DatasetType.RIO10, dir_dataset,  name_train_sequence, name_test_sequence)

    intrinsics_train, intrinsics_test = dataset_manager.read_intrinsics() # identical
    # print(intrinsics_train.intrinsic_matrix)
    # print(intrinsics_test.intrinsic_matrix)
    # NOTE: read data
    dataset = dataset_manager.read_data(is_NeuralRouting_normalized=False)
    IDX_TRAIN = 277
    IDX_TEST = 817

    



    return