import numpy as np

def get_data(n: int =100, low: int = 0, high: int = 25):
    """
    Return an nx2 array of random integers between low and high (inclusive)
    """
    
    return np.random.randint(low = low, high = high + 1, size = (n,2))