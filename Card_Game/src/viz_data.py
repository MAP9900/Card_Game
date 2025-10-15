import os
import numpy as np
import matplotlib.pyplot as plt
from src.gen_data import get_decks
from src.score_data import p2_win_prob_matrix, p2_win_prob_from_mats, score_humble_nishiyama
from src.utils import time_and_size


def _default_fig_dir() -> str:
    """
    Helper function to create/find file path for saved figures
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

@time_and_size
def save_p2_win_prob_heatmap_from_counts(win_counts: np.ndarray, tie_counts: np.ndarray, total_decks: int,*,
                                        out_dir: str | None = None, filename: str | None = None,
                                        title: str | None = None,) -> str:
    """
    Save a P2 win probability heatmap using aggregated win/tie counts instead of raw matrices.
    This avoids materializing all score matrices when the deck count is extremely large.
    """
    
    if out_dir is None:
        out_dir = _default_fig_dir()
    if filename is None:
        filename = "humble_nishiyama_p2_win_prob_from_counts.png"
    if title is None:
        title = f"Humble–Nishiyama P2 Win Probabilities (n={total_decks})"

    win_probs = win_counts.astype(np.float64) / float(total_decks)
    tie_probs = tie_counts.astype(np.float64) / float(total_decks)

    diag_mask = np.eye(win_counts.shape[0], dtype=bool)
    win_probs[diag_mask] = np.nan
    tie_probs[diag_mask] = np.nan

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgray')

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(
        np.ma.masked_invalid(win_probs), vmin=0.0, vmax=1.0,
        cmap=cmap, interpolation='nearest')
    plt.colorbar(im, label='P2 win probability (per deck)')

    ticks = list(range(8))
    labels = [format(i, '03b') for i in range(8)]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('P2 Pattern (000 - 111)')
    plt.ylabel('P1 Pattern (000 - 111)')
    plt.title(title)

    for i in range(8):
        for j in range(8):
            if i == j or np.isnan(win_probs[i, j]):
                continue
            win_pct = int(round(win_probs[i, j] * 100))
            tie_pct = int(round(tie_probs[i, j] * 100)) if not np.isnan(tie_probs[i, j]) else 0
            plt.text(j, i, f"{win_pct}({tie_pct})", ha='center', va='center', color='black', fontsize=8)

    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path




#Extra Viz Functions

#Test heatmap function
# @time_and_size
# def save_hn_score_heatmap(deck_seed: int = 42, out_dir: str | None = None, filename: str | None = None) -> str:
#     """
#     Compute the Humble–Nishiyama 8x8 score matrix for a single deck and save a heatmap. 
#     Used for testing to validate scoring functions. 
#     Returns the output file path.
#     """

#     if out_dir is None:
#         out_dir = _default_fig_dir()
#     if filename is None:
#         filename = f"humble_nishiyama_seed{deck_seed}.png"

#     deck = get_decks(1, deck_seed)[0]
#     m = score_humble_nishiyama(deck)

#     m_plot = m.astype(float).copy()
#     np.fill_diagonal(m_plot, np.nan)

#     cmap = plt.cm.viridis.copy()
#     cmap.set_bad(color='lightgray')

#     plt.figure(figsize=(6, 5))
#     im = plt.imshow(np.ma.masked_invalid(m_plot), cmap=cmap, interpolation='nearest')
#     plt.colorbar(im, label='P1 score')

#     ticks = list(range(8))
#     labels = [format(i, '03b') for i in range(8)]
#     plt.xticks(ticks, labels)
#     plt.yticks(ticks, labels)
#     plt.xlabel('P2 Pattern (000 - 111)')
#     plt.ylabel('P1 Pattern (000 - 111)')
#     plt.title(f'Humble–Nishiyama Score Heatmap (seed={deck_seed})')

#     out_path = os.path.join(out_dir, filename)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     return out_path


# @time_and_size
# def save_p2_win_prob_heatmap_from_mats(mats: np.ndarray, out_dir: str | None = None, filename: str | None = None) -> str:
#     """
#     Save a P2 win probability heatmap computed from precomputed HN matrices (mats)
#     mats should have shape (n, 8, 8). 
#     """
#     #Determine Output path and file name. Has presets if none are given
#     if out_dir is None:
#         out_dir = _default_fig_dir()
#     if filename is None:
#         filename = "humble_nishiyama_p2_win_prob_from_mats.png"

#     win_probs, tie_probs = p2_win_prob_from_mats(mats, return_ties=True)
#     cmap = plt.cm.plasma.copy() #Color = plasma
#     cmap.set_bad(color='lightgray')
#     plt.figure(figsize=(6.5, 5.5))
#     im = plt.imshow(np.ma.masked_invalid(win_probs), vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')
#     plt.colorbar(im, label='P2 win probability (per deck)')
#     ticks = list(range(8))
#     labels = [format(i, '03b') for i in range(8)]
#     plt.xticks(ticks, labels)
#     plt.yticks(ticks, labels)
#     plt.xlabel('P2 Pattern (000 - 111)')
#     plt.ylabel('P1 Pattern (000 - 111)')
#     plt.title('Humble–Nishiyama P2 Win Probabilities (from saved scores)')
#     for i in range(8):
#         for j in range(8):
#             win_val = win_probs[i, j]
#             if i == j or np.isnan(win_val):
#                 continue
#             color = 'black' 
#             tie_val = tie_probs[i, j]
#             win_pct = int(round(win_val * 100))
#             tie_pct = int(round(tie_val * 100)) if not np.isnan(tie_val) else 0
#             plt.text(j, i, f"{win_pct}({tie_pct})", ha='center', va='center', color=color, fontsize=8)
#     out_path = os.path.join(out_dir, filename)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     return out_path
