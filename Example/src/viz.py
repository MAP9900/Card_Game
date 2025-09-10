import matplotlib.pyplot as plt 
import numpy as np
import os

PATH_FIGURES = 'figures'
def make_figure(data: np.ndarray, filename: str) -> None:
    """
    Inputs: data 2xN np.ndarray, filename: filename for the figure (will be stored in figures/)
    """

    full_filename = os.path.join(PATH_FIGURES, filename)
    if not os.path.exists(PATH_FIGURES):
        os.mkdir(PATH_FIGURES)
    if os.path.exists(full_filename):
            raise FileExistsError(f'{full_filename} already exists, select a new name!')

    plt.figure(figsize = (5, 5))
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(full_filename, bbox_inches='tight')
    plt.show()
    return None