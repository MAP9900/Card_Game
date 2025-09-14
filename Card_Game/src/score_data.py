import numpy as np
from typing import Tuple
from src.gen_data import generate_seeds, deck_from_seed


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


def score_humble_nishiyama(deck: np.ndarray) -> np.ndarray:
    """
    Scores 8x8 matrix of all valid player matchups (P1 != P2 only).
    Each player picks a 3-card pattern (000–111); flip the deck and score
    one trick when a match occurs. Diagonal (same-pattern) is invalid (-1).
    Returns scores[i, j] = P1's score when P1 uses i, P2 uses j.
    """
    assert deck.shape == (52,)
    assert np.sum(deck) == 26

    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    scores = np.full((8, 8), -1, dtype=np.int16)

    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i == j:
                continue
            score = 0
            idx = 0
            while idx <= 49:
                window = deck[idx:idx+3]
                if np.array_equal(window, p1):
                    score += 1
                    idx += 3
                elif np.array_equal(window, p2):
                    idx += 3
                else:
                    idx += 1
            scores[i, j] = score
    return scores


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


def p2_win_prob_from_mats(mats: np.ndarray) -> np.ndarray:
    """
    Compute P2 win probabilities from a batch of Humble–Nishiyama matrices.
    `mats` shape should be (n_decks, 8, 8), where mats[d, i, j] is P1's score
    on deck d when P1 uses pattern i and P2 uses pattern j.

    Returns an 8x8 float array with diagonal set to NaN, where entry [i, j]
    is the fraction of decks for which P2's score > P1's score when P1 picks i
    and P2 picks j. P2's score on a deck for (i, j) equals mats[d, j, i].
    """
    if mats.ndim != 3 or mats.shape[1:] != (8, 8):
        raise ValueError("mats must have shape (n, 8, 8)")
    n = mats.shape[0]
    p2_wins = (mats[:, :, :].transpose(0, 2, 1) > mats).sum(axis=0)  # count decks where mats[:, j, i] > mats[:, i, j]
    probs = p2_wins / float(n)
    probs[np.eye(8, dtype=bool)] = np.nan
    return probs
