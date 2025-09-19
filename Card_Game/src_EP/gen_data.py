import numpy as np
import os
from utils import time_and_size


def generate_seeds(n: int, base_seed = 12345) -> np.ndarray:
    """
    Return n independent uint64 seeds based off set base seed
    """
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)

def deck_from_seed(seed: int) -> np.ndarray:
    """
    Return a 52-card deck with 26 zeros (B) and 26 ones (R), shuffled by generated seeds
    """
    rng = np.random.default_rng(seed)
    deck = np.empty(52, dtype=np.uint8)
    deck[:26] = 0
    deck[26:] = 1
    rng.shuffle(deck)
    return deck


def compute_scores_from_seeds(seeds: np.ndarray, score_fn) -> np.ndarray:
    """
    Rebuilds each deck from the generated seeds and the applies a scoring function to the decks. 
    Supports score functions that return either:
      - a scalar (int/float), or
      - a NumPy array of any shape
         
    Returns an array shaped as:
      - (len(seeds),) for scalar scores
      - (len(seeds), *score_shape) for array scores
    """
    n = len(seeds)
    if n == 0:
        return np.array([], dtype=np.int16)

    #Probe first result
    first = score_fn(deck_from_seed(int(seeds[0])))
    if np.isscalar(first):
        out = np.empty(n, dtype=np.asarray(first).dtype)
        out[0] = first
        for i in range(1, n):
            out[i] = score_fn(deck_from_seed(int(seeds[i])))
        return out
    else:
        first_arr = np.asarray(first)
        out = np.empty((n, *first_arr.shape), dtype=first_arr.dtype)
        out[0] = first_arr
        for i in range(1, n):
            out[i] = np.asarray(score_fn(deck_from_seed(int(seeds[i]))))
        return out


def _data_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@time_and_size
def save_seeds(seeds: np.ndarray, filename: str = "seeds.npy") -> str:
    """Save seeds to Card_Game/data and return the saved path."""
    path = os.path.join(_data_dir(), filename)
    np.save(path, seeds)
    return path


def load_seeds(filename: str = "seeds.npy") -> np.ndarray:
    """Load seeds from Card_Game/data and return the array."""
    path = os.path.join(_data_dir(), filename)
    return np.load(path)


@time_and_size
def save_scores(scores: np.ndarray, filename: str = "scores.npy") -> str:
    """Save scores/matrices to Card_Game/data and return the saved path."""
    path = os.path.join(_data_dir(), filename)
    np.save(path, scores)
    return path


def load_scores(filename: str = "scores.npy") -> np.ndarray:
    """Load scores/matrices from Card_Game/data and return the array."""
    path = os.path.join(_data_dir(), filename)
    return np.load(path)

if __name__ == "__main__":
    # Generate some example seeds
    seeds = generate_seeds(10)
    print("Generated Seeds:", seeds)

    # Save seeds and measure time
    save_seeds(seeds)

    # Compute example scores
    def example_score_fn(deck):
        return np.sum(deck)  # Example scoring function: sum of the deck (number of 1s)

    scores = compute_scores_from_seeds(seeds, example_score_fn)
    print("Computed Scores:", scores)

    # Save scores and measure time
    save_scores(scores)

## score_humble_nishiyama moved to src/score_data.py; imported above for callers
