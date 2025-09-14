import numpy as np
import os
from src.gen_data import generate_seeds, compute_scores_from_seeds, save_seeds, load_seeds, save_scores,load_scores
from src.score_data import score_humble_nishiyama
from src.viz_data import save_hn_score_heatmap, save_p2_win_prob_heatmap, save_p2_win_prob_heatmap_from_mats

def simple_score(deck: np.ndarray) -> int:
    # Example: sum of the first 10 cards (varies by seed)
    return int(deck[:10].sum())

def main():
    seeds = generate_seeds(5, base_seed=1976)
    scores = compute_scores_from_seeds(seeds, simple_score)
    print("seeds:", seeds)
    print("scores:", scores)


if __name__ == "__main__":
    # Quick sanity: scalar scores across a few decks
    seeds = generate_seeds(5, base_seed=1976)
    scores = compute_scores_from_seeds(seeds, simple_score)
    print("seeds:", seeds)
    print("scores:", scores)

    # Single-deck HN score heatmap
    out1 = save_hn_score_heatmap(deck_seed=42)
    print(f"Saved score heatmap to: {out1}")

    # Empirical P2 win probability heatmap (fresh compute)
    out2 = save_p2_win_prob_heatmap(n_games=100, base_seed=2024)
    print(f"Saved probability heatmap to: {out2}")

    # Full pipeline with caching: 1000 decks -> save seeds/scores -> viz from saved
    n = 1000
    seeds_file = f"seeds_{n}.npy"
    scores_file = f"scores_{n}.npy"

    #Seeds
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, seeds_file)):
        seeds_big = load_seeds(seeds_file)
        print(f"Loaded seeds from data/{seeds_file}")
    else:
        seeds_big = generate_seeds(n, base_seed=2024)
        save_seeds(seeds_big, seeds_file)

    # Scores (per-deck 8x8 matrices)
    if os.path.exists(os.path.join(data_dir, scores_file)):
        mats = load_scores(scores_file)
        print(f"Loaded scores from data/{scores_file}")
    else:
        mats = compute_scores_from_seeds(seeds_big, score_humble_nishiyama)
        save_scores(mats, scores_file)

    out3 = save_p2_win_prob_heatmap_from_mats(mats)
    print(f"Saved combined probability heatmap to: {out3}")
