import numpy as np
import os
from src.utils import time_and_size

PATH_DATA = "/Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data"
HALF_DECK_SIZE = 26

@time_and_size
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






# def generate_seeds(n: int, base_seed = 12345) -> np.ndarray:
#     """
#     Return n independent uint64 seeds based off set base seed
#     """
#     rng = np.random.default_rng(base_seed)
#     return rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)

# def deck_from_seed(seed: int) -> np.ndarray:
#     """
#     Return a 52-card deck with 26 zeros (B) and 26 ones (R), shuffled by generated seeds
#     """
#     rng = np.random.default_rng(seed)
#     deck = np.empty(52, dtype=np.uint8)
#     deck[:26] = 0
#     deck[26:] = 1
#     rng.shuffle(deck)
#     return deck


# def compute_scores_from_seeds(seeds: np.ndarray, score_fn) -> np.ndarray:
#     """
#     Rebuilds each deck from the generated seeds and the applies a scoring function to the decks. 
#     Supports score functions that return either:
#       - a scalar (int/float), or
#       - a NumPy array of any shape
         
#     Returns an array shaped as:
#       - (len(seeds),) for scalar scores
#       - (len(seeds), *score_shape) for array scores
#     """
#     n = len(seeds)
#     if n == 0:
#         return np.array([], dtype=np.int16)

#     #Probe first result
#     first = score_fn(deck_from_seed(int(seeds[0])))
#     if np.isscalar(first):
#         out = np.empty(n, dtype=np.asarray(first).dtype)
#         out[0] = first
#         for i in range(1, n):
#             out[i] = score_fn(deck_from_seed(int(seeds[i])))
#         return out
#     else:
#         first_arr = np.asarray(first)
#         out = np.empty((n, *first_arr.shape), dtype=first_arr.dtype)
#         out[0] = first_arr
#         for i in range(1, n):
#             out[i] = np.asarray(score_fn(deck_from_seed(int(seeds[i]))))
#         return out


# def _data_dir() -> str:
#     base_dir = os.path.dirname(os.path.dirname(__file__))
#     data_dir = os.path.join(base_dir, "data")
#     os.makedirs(data_dir, exist_ok=True)
#     return data_dir


# @time_and_size
# def save_seeds(seeds: np.ndarray, filename: str = "seeds.npy") -> str:
#     """Save seeds to Card_Game/data and return the saved path."""
#     path = os.path.join(_data_dir(), filename)
#     np.save(path, seeds)
#     return path


# def load_seeds(filename: str = "seeds.npy") -> np.ndarray:
#     """Load seeds from Card_Game/data and return the array."""
#     path = os.path.join(_data_dir(), filename)
#     return np.load(path)


# @time_and_size
# def save_scores(scores: np.ndarray, filename: str = "scores.npy") -> str:
#     """Save scores/matrices to Card_Game/data and return the saved path."""
#     path = os.path.join(_data_dir(), filename)
#     np.save(path, scores)
#     return path


# def load_scores(filename: str = "scores.npy") -> np.ndarray:
#     """Load scores/matrices from Card_Game/data and return the array."""
#     path = os.path.join(_data_dir(), filename)
#     return np.load(path)


# score_humble_nishiyama moved to src/score_data.py. Now imported above
