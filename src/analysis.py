#A file like this is should only contain code relevant to data processing
import os
import numpy as np

PATH_DATA = 'data/'


def make_data(n: int =100, low: int = 0, high: int = 25):
    """
    Return an nx2 array of random integers between low and high (inclusive)
    """
    
    return np.random.randint(low = low, high = high + 1, size = (n,2))

def save_data(data: np.ndarray, filename: str) -> None:
    """
    Save a numpy array in the defualt output directory , esnuring the output directory exsits. 
    """

    full_filename = os.path.join(PATH_DATA, filename)
    if not os.path.exists(PATH_DATA):
        os.mkdir(PATH_DATA)
    if type(data) != np.ndarray:
        raise TypeError(f'Data should be np.ndarray {type(data)}')
    else: 
        if os.path.exists(full_filename):
            raise FileExistsError(f'{full_filename} already exists, select a new name!')
        np.save(os.path.join(PATH_DATA, filename), data)

    return None #return same as return None

def load_data(filename: str) -> np.ndarray:
    """
    Loads data from an .npy file located in the default directory
    """

    return np.load(os.path.join(PATH_DATA, filename))