import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _default_fig_dir() -> str:
    """
    Helper function to create/find file path for saved figures
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

# @time_and_size
def save_p2_win_prob_heatmap_from_counts(win_counts: np.ndarray, tie_counts: np.ndarray, total_decks: int,*,
                                        out_dir: str | None = None, filename: str | None = None,
                                        title: str | None = None,) -> str:
    """
    Save a P2 win probability heatmap using aggregated win/tie counts instead of raw matrices.
    This avoids materializing all score matrices when the deck count is extremely large.
    """
    
    if out_dir is None:
        out_dir = _default_fig_dir()

    win_probs = win_counts.astype(np.float64) / float(total_decks)
    tie_probs = tie_counts.astype(np.float64) / float(total_decks)

    diag_mask = np.eye(win_counts.shape[0], dtype=bool)
    win_probs[diag_mask] = np.nan
    tie_probs[diag_mask] = np.nan

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='lightgray')

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(np.ma.masked_invalid(win_probs), vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')

    ticks = list(range(8))
    labels = [format(i, '03b') for i in range(8)]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('My Choice')
    plt.ylabel('Opponent choice')
    plt.title(title)
    ax = plt.gca()

    #Highlight best cell in each row
    for i in range(win_probs.shape[0]):
        row = win_probs[i]
        if np.all(np.isnan(row)):
            continue
        max_val = np.nanmax(row)
        if np.isnan(max_val):
            continue
        max_cols = np.where(np.isclose(row, max_val, rtol=1e-9, atol=1e-12))[0]
        for j in max_cols:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.8, zorder=3))

    #Add Dack Combination indices 
    for i in range(8):
        for j in range(8):
            if i == j or np.isnan(win_probs[i, j]):
                continue
            win_pct = int(round(win_probs[i, j] * 100))
            tie_pct = int(round(tie_probs[i, j] * 100)) if not np.isnan(tie_probs[i, j]) else 0
            plt.text(j, i, f"{win_pct}({tie_pct})", ha='center', va='center', color='black', fontsize=8)
    for spine in plt.gca().spines.values():
       spine.set_visible(False)
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
