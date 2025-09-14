import os
import numpy as np

from src.gen_data import (
    generate_seeds,
    save_seeds,
    load_seeds,
    compute_scores_from_seeds,
    save_scores,
    load_scores,
)
from src.score_data import score_humble_nishiyama
from src.viz_data import save_p2_win_prob_heatmap_from_mats


def data_dir() -> str:
    base_dir = os.path.dirname(__file__)
    d = os.path.join(base_dir, "data")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    n = 1000
    seeds_file = f"seeds_{n}.npy"
    scores_file = f"scores_{n}.npy"

    # 1) Generate or load seeds
    seeds_path = os.path.join(data_dir(), seeds_file)
    if os.path.exists(seeds_path):
        print(f"Loading existing seeds: {seeds_path}")
        seeds = load_seeds(seeds_file)
    else:
        seeds = generate_seeds(n, base_seed=2024)
        save_seeds(seeds, seeds_file)

    # 2) Compute or load per-deck HN matrices
    scores_path = os.path.join(data_dir(), scores_file)
    if os.path.exists(scores_path):
        print(f"Loading existing scores: {scores_path}")
        mats = load_scores(scores_file)
    else:
        mats = compute_scores_from_seeds(seeds, score_humble_nishiyama)
        save_scores(mats, scores_file)

    # 3) Visualize P2 win probabilities from saved matrices
    out_fig = save_p2_win_prob_heatmap_from_mats(mats)
    print(f"Saved combined probability heatmap to: {out_fig}")


if __name__ == "__main__":
    main()
