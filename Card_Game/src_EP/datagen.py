import numpy as np
import os
from helpers import PATH_DATA, debugger_factory 
from utils import time_and_size

HALF_DECK_SIZE = 26

debug = debugger_factory(show_args=True)

@debug
def get_decks(n_decks: int, 
              seed: int, 
              half_deck_size: int = HALF_DECK_SIZE) -> np.ndarray:
    """
    Efficiently generate `n_decks` shuffled decks using NumPy.
    
    Args:
        n_decks (int): Number of decks to generate.
        seed (int): Random seed for reproducibility.
        half_deck_size (int): Number of cards of each color in a half deck.
    
    Returns:
        np.ndarray: 2D array of shape (n_decks, num_cards), each row is a shuffled deck.
    """
    init_deck = [0] * half_deck_size + [1] * half_deck_size
    decks = np.tile(init_deck, (n_decks, 1))
    rng = np.random.default_rng(seed)
    rng.permuted(decks, axis=1, out=decks)
    return decks

@time_and_size
def save_decks(decks: np.ndarray, 
               seed: int, 
               batch_size: int = 100_000,
               filename: str = "decks_batch.npy"):
    """
    Saves decks and the seed used to PATH_DATA.
    """
    os.makedirs(PATH_DATA, exist_ok=True)
    saved_files = []
    num_batches = (len(decks) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch = decks[i * batch_size:(i + 1) * batch_size]
        batch_filename = f"{filename}_{i}.npy"
        batch_path = os.path.join(PATH_DATA, batch_filename)
        np.save(batch_path, batch)
        print(f"Saved chunk {i + 1}/{num_batches} to {batch_path}")
        saved_files.append(batch_path)
    
    # Save the seed for reproducibility
    seed_file = os.path.join(PATH_DATA, "decks_seed.npy")
    np.save(seed_file, np.array([seed], dtype=np.uint64))
    print(f"Saved seed to {seed_file}")
    saved_files.append(seed_file)
    return saved_files

def load_decks(filename: str = "decks.npy"):
    """
    Loads decks and seed from PATH_DATA.
    """
    path = os.path.join(PATH_DATA, filename)
    decks = np.load(path)
    seed_file = os.path.join(PATH_DATA, "decks_seed.npy")
    seed = int(np.load(seed_file)[0])
    print(f"Loaded decks from {path} with seed {seed}")
    return decks, seed

if __name__ == "__main__":
    N_DECKS = 2_000_000
    SEED = 12345 

    decks = get_decks(N_DECKS, SEED)
    save_decks(decks, SEED)
