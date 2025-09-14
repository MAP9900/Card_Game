import numpy as np
from typing import Tuple

from .gen_data import generate_seeds, deck_from_seed


def _count_both(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    """
    Count how many tricks P1 and P2 take on a single deck given their 3-bit patterns.
    Returns (p1_count, p2_count).
    """
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


def p2_win_prob_matrix(n_games: int = 100, base_seed: int = 2024) -> np.ndarray:
    """
    Estimate P2's win probability per matchup (P1 pattern i, P2 pattern j)
    across `n_games` random decks. A win is counted when P2's total tricks > P1's.

    Returns an 8x8 float array with diagonal set to NaN (invalid same-pattern).
    """
    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    seeds = generate_seeds(n_games, base_seed=base_seed)

    win_sum = np.zeros((8, 8), dtype=int)
    game_cnt = np.zeros((8, 8), dtype=int)

    for s in seeds:
        deck = deck_from_seed(int(s))
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                p1c, p2c = _count_both(deck, patterns[i], patterns[j])
                game_cnt[i, j] += 1
                if p2c > p1c:
                    win_sum[i, j] += 1

    with np.errstate(invalid='ignore'):
        probs = win_sum / np.maximum(game_cnt, 1)
    probs[np.eye(8, dtype=bool)] = np.nan
    return probs


def p2_win_prob_row(p1_index: int, n_games: int = 100, base_seed: int = 2024) -> np.ndarray:
    """
    Estimate P2 win probabilities for a fixed P1 pattern `p1_index` vs all P2 patterns.
    Returns a length-8 float array with NaN at the diagonal position.
    """
    if not (0 <= p1_index <= 7):
        raise ValueError("p1_index must be between 0 and 7 inclusive")

    full = p2_win_prob_matrix(n_games=n_games, base_seed=base_seed)
    row = full[p1_index].copy()
    return row

