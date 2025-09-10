import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../.."))) #Allow for imports from src
from src.gen_data import generate_seeds, compute_scores_from_seeds, deck_from_seed

def simple_score(deck: np.ndarray) -> int:
    # Example: sum of the first 10 cards (varies by seed)
    return int(deck[:10].sum())

def main():
    seeds = generate_seeds(5, base_seed=1976)
    scores = compute_scores_from_seeds(seeds, simple_score)
    print("seeds:", seeds)
    print("scores:", scores)


if __name__ == "__main__":
    from src.gen_data import compute_scores_from_seeds, deck_from_seed

    seeds = generate_seeds(5, base_seed=1976)
    scores = compute_scores_from_seeds(seeds, simple_score)
    print("seeds:", seeds)
    print("scores:", scores)


