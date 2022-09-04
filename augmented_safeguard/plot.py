

import numpy as np
from scipy.spatial.transform import Rotation
import quaternion

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_rotation_variance(R_gt, list_R_targets):

    print(f"R_gt = \n{R_gt}")
    R_mean = np.asarray(list_R_targets).mean(axis=0)
    print(f"R_mean = \n{R_mean}")
    # exit()




    return