import os
import numpy as np
import matplotlib.pyplot as plt
from src.gen_data import deck_from_seed
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

#Test heatmap function
@time_and_size
def save_hn_score_heatmap(deck_seed: int = 42, out_dir: str | None = None, filename: str | None = None) -> str:
    """
    Compute the Humble–Nishiyama 8x8 score matrix for a single deck and save a heatmap. 
    Used for testing to validate scoring functions. 
    Returns the output file path.
    """

    if out_dir is None:
        out_dir = _default_fig_dir()
    if filename is None:
        filename = f"humble_nishiyama_seed{deck_seed}.png"

    deck = deck_from_seed(deck_seed)
    m = score_humble_nishiyama(deck)

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
    plt.xlabel('P2 Pattern (000 - 111)')
    plt.ylabel('P1 Pattern (000 - 111)')
    plt.title(f'Humble–Nishiyama Score Heatmap (seed={deck_seed})')

    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


@time_and_size
def save_p2_win_prob_heatmap_from_mats(mats: np.ndarray, out_dir: str | None = None, filename: str | None = None) -> str:
    """
    Save a P2 win probability heatmap computed from precomputed HN matrices (mats)
    mats should have shape (n, 8, 8). 
    """
    #Determine Output path and file name. Has presets if none are given
    if out_dir is None:
        out_dir = _default_fig_dir()
    if filename is None:
        filename = "humble_nishiyama_p2_win_prob_from_mats.png"

    win_probs, tie_probs = p2_win_prob_from_mats(mats, return_ties=True)
    cmap = plt.cm.plasma.copy() #Color = plasma
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(np.ma.masked_invalid(win_probs), vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, label='P2 win probability (per deck)')
    ticks = list(range(8))
    labels = [format(i, '03b') for i in range(8)]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('P2 Pattern (000 - 111)')
    plt.ylabel('P1 Pattern (000 - 111)')
    plt.title('Humble–Nishiyama P2 Win Probabilities (from saved scores)')
    for i in range(8):
        for j in range(8):
            win_val = win_probs[i, j]
            if i == j or np.isnan(win_val):
                continue
            color = 'black' 
            tie_val = tie_probs[i, j]
            win_pct = int(round(win_val * 100))
            tie_pct = int(round(tie_val * 100)) if not np.isnan(tie_val) else 0
            plt.text(j, i, f"{win_pct}({tie_pct})", ha='center', va='center', color=color, fontsize=8)
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


#Extra Viz Functions

# #Runs monte carlo sims on new games. (n_games = ...) Used to regenerate probabilties on the fly. 
# @time_and_size
# def save_p2_win_prob_heatmap(n_games: int = 100, base_seed: int = 2024, out_dir: str | None = None, filename: str | None = None) -> str:
#     """
#     Estimate P2's win probability per matchup across many decks and save a heatmap.
#     Returns the output file path.
#     """

#     if out_dir is None:
#         out_dir = _default_fig_dir()
#     if filename is None:
#         filename = f"humble_nishiyama_p2_win_prob_{n_games}.png"

#     probs = p2_win_prob_matrix(n_games=n_games, base_seed=base_seed)

#     cmap = plt.cm.viridis.copy()
#     cmap.set_bad(color='lightgray')

#     plt.figure(figsize=(6.5, 5.5))
#     im = plt.imshow(np.ma.masked_invalid(probs), vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')
#     plt.colorbar(im, label='P2 win probability (per deck)')

#     ticks = list(range(8))
#     labels = [format(i, '03b') for i in range(8)]
#     plt.xticks(ticks, labels)
#     plt.yticks(ticks, labels)
#     plt.xlabel('P2 Pattern (000 - 111)')
#     plt.ylabel('P1 Pattern (000 - 111)')
#     plt.title(f'Humble–Nishiyama P2 Win Probabilities (n={n_games})')

#     #Add win percentages to the heatmap
#     for i in range(8):
#         for j in range(8):
#             val = probs[i, j]
#             if i == j or np.isnan(val):
#                 continue
#             color = 'black'
#             plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
#     #Save figure
#     out_path = os.path.join(out_dir, filename)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     return out_path


#Second Viz Function for displaying P1’s total tricks vs every P2 pattern on one specific shuffled deck (deterministic for the given seed) 

# @time_and_size
# def save_hn_score_heatmap(deck_seed: int = 42, out_dir: str | None = None, filename: str | None = None) -> str:
#     """
#     Compute the Humble–Nishiyama 8x8 score matrix for a single deck and save a heatmap.
#     Returns the output file path.
#     """

#     if out_dir is None:
#         out_dir = _default_fig_dir()
#     if filename is None:
#         filename = f"humble_nishiyama_seed{deck_seed}.png"

#     deck = deck_from_seed(deck_seed) #Single Deck's seed to display results 
#     m = score_humble_nishiyama(deck) #Scoring Method

#     m_plot = m.astype(float).copy()
#     np.fill_diagonal(m_plot, np.nan) #Mask diagonal since no results for same sequences 

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
#     plt.title(f'Humble–Nishiyama Score Heatmap (seed={deck_seed})') #Note results are for a single seed. Still in testing...
#     #Save figure
#     out_path = os.path.join(out_dir, filename)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     return out_path
