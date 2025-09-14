import numpy as np
import os, sys
from src.gen_data import generate_seeds,compute_scores_from_seeds,deck_from_seed,score_humble_nishiyama
from src.utils import track_perf
from src.viz_data import save_hn_score_heatmap,save_p2_win_prob_heatmap
from src.score_data import p2_win_prob_matrix, p2_win_prob_row
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../.."))) #Allow for imports from src

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

    # --- Humble–Nishiyama single-deck score heatmap ---
    out1 = save_hn_score_heatmap(deck_seed=42)
    print(f"Saved score heatmap to: {out1}")

    # --- Empirical P2 win probability heatmap across many decks ---
    out2 = save_p2_win_prob_heatmap(n_games=100, base_seed=2024)
    print(f"Saved probability heatmap to: {out2}")
