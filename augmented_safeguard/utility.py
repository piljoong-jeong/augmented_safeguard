import copy

import numpy as np

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def uniform_sample(arr, n_stripes):
    """
    uniform sampling
    
    """


    result = copy.deepcopy(arr)
    indices = [
        i + n_stripes*j 
        for i in range(n_stripes) 
        for j in range(3)
    ]

    return result[indices]
    