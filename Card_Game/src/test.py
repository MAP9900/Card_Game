import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../.."))) #Allow for imports from src
from src.gen_data import (
    generate_seeds,
    compute_scores_from_seeds,
    deck_from_seed,
    score_humble_nishiyama,
)

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

    # --- Humble-Nishiyama 8x8 test and heatmap ---
    # Build one deck, compute 8x8 matrix, and save a heatmap figure
    deck = deck_from_seed(42)
    m = score_humble_nishiyama(deck)
    print("humble_nishiyama matrix shape:", m.shape)

    # Create figures/ if needed in Card_Game root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot heatmap using matplotlib
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Mask diagonal (-1 invalid matchups)
        m_plot = m.astype(float).copy()
        np.fill_diagonal(m_plot, np.nan)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')

        plt.figure(figsize=(6, 5))
        im = plt.imshow(np.ma.masked_invalid(m_plot), cmap=cmap, interpolation='nearest')
        plt.colorbar(im, label='P1 score')

        ticks = list(range(8))
        labels = [format(i, '03b') for i in range(8)]
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)
        plt.xlabel('P2 pattern (000..111)')
        plt.ylabel('P1 pattern (000..111)')
        plt.title('Humble–Nishiyama Score Heatmap (seed=42)')

        out_path = os.path.join(fig_dir, 'humble_nishiyama_seed42.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved heatmap to: {out_path}")
        # plt.show()  # optional
    except ImportError:
        print("matplotlib not installed; skipping heatmap. Install with: uv add matplotlib")

    # --- Empirical P2 WIN probability heatmap over many games ---
    # For each (P1 pattern i, P2 pattern j), estimate probability
    # that P2 strictly wins the game under Humble–Nishiyama rules
    # (i.e., p2_count > p1_count on a deck). Ties count as 0.
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        def count_both(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[int, int]:
            p1c = 0
            p2c = 0
            idx = 0
            while idx <= 49:
                window = deck[idx:idx+3]
                if np.array_equal(window, p1):
                    p1c += 1
                    idx += 3
                elif np.array_equal(window, p2):
                    p2c += 1
                    idx += 3
                else:
                    idx += 1
            return p1c, p2c

        patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
        n_games = 100
        seeds_prob = generate_seeds(n_games, base_seed=2024)

        # Accumulate strict-win counts and total trials per matchup
        win_sum = np.zeros((8, 8), dtype=int)
        game_cnt = np.zeros((8, 8), dtype=int)

        for s in seeds_prob:
            deck = deck_from_seed(int(s))
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    p1c, p2c = count_both(deck, patterns[i], patterns[j])
                    # Count a trial; add a win if P2 strictly wins
                    game_cnt[i, j] += 1
                    if p2c > p1c:
                        win_sum[i, j] += 1

        with np.errstate(invalid='ignore'):
            probs = win_sum / np.maximum(game_cnt, 1)
        probs[np.eye(8, dtype=bool)] = np.nan

        base_dir = os.path.dirname(os.path.dirname(__file__))
        fig_dir = os.path.join(base_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(np.ma.masked_invalid(probs), vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, label='P2 win probability (per deck)')

        ticks = list(range(8))
        labels = [format(i, '03b') for i in range(8)]
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)
        plt.xlabel('P2 pattern (000..111)')
        plt.ylabel('P1 pattern (000..111)')
        plt.title(f'Humble–Nishiyama P2 Win Probabilities (n={n_games})')

        # Add numerical annotations
        for i in range(8):
            for j in range(8):
                if i == j or game_cnt[i, j] == 0:
                    continue
                val = probs[i, j]
                txt_color = 'white' if val > 0.6 else 'black'
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=txt_color, fontsize=8)

        out_probs = os.path.join(fig_dir, f'humble_nishiyama_p2_win_prob_{n_games}.png')
        plt.tight_layout()
        plt.savefig(out_probs, dpi=150)
        print(f"Saved probability heatmap to: {out_probs}")
        # plt.show()  # optional
    except ImportError:
        print("matplotlib not installed; skipping probability heatmap. Install with: uv add matplotlib")
