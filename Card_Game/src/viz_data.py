import os
import numpy as np

from .gen_data import deck_from_seed, score_humble_nishiyama
from .score_data import p2_win_prob_matrix
from .utils import time_and_size


def _default_fig_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


@time_and_size
def save_hn_score_heatmap(deck_seed: int = 42, out_dir: str | None = None, filename: str | None = None) -> str:
    """
    Compute the Humble–Nishiyama 8x8 score matrix for a single deck and save a heatmap.
    Returns the output file path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib not installed; install with 'uv add matplotlib'") from e

    if out_dir is None:
        out_dir = _default_fig_dir()
    if filename is None:
        filename = f"humble_nishiyama_seed{deck_seed}.png"

    deck = deck_from_seed(deck_seed)
    m = score_humble_nishiyama(deck)

    # Mask diagonal (invalid same-pattern)
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
    plt.title(f'Humble–Nishiyama Score Heatmap (seed={deck_seed})')

    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


@time_and_size
def save_p2_win_prob_heatmap(n_games: int = 100, base_seed: int = 2024, out_dir: str | None = None, filename: str | None = None) -> str:
    """
    Estimate P2's win probability per matchup across many decks and save a heatmap.
    Returns the output file path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib not installed; install with 'uv add matplotlib'") from e

    if out_dir is None:
        out_dir = _default_fig_dir()
    if filename is None:
        filename = f"humble_nishiyama_p2_win_prob_{n_games}.png"

    probs = p2_win_prob_matrix(n_games=n_games, base_seed=base_seed)

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

    # Add numeric annotations
    for i in range(8):
        for j in range(8):
            val = probs[i, j]
            if i == j or np.isnan(val):
                continue
            txt_color = 'white' if val > 0.6 else 'black'
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=txt_color, fontsize=8)

    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
