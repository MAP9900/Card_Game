import numpy as np
from typing import Tuple
from src.gen_data import get_decks

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


def _count_cards(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    """Return card totals awarded to each player using the "pot" scoring variant."""
    p1_cards = 0
    p2_cards = 0
    pot_start = 0

    for idx in range(deck.shape[0]):
        pot_size = idx - pot_start + 1
        if pot_size < 3:
            continue
        window = deck[idx-2:idx+1]
        if np.array_equal(window, p1):
            p1_cards += pot_size
            pot_start = idx + 1
        elif np.array_equal(window, p2):
            p2_cards += pot_size
            pot_start = idx + 1

    return p1_cards, p2_cards


def score_humble_nishiyama(deck: np.ndarray, *, return_ties: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Scores 8x8 matrix of all valid player matchups (P1 != P2 only).
    Each player picks a 3-card pattern (000–111); flip the deck and score
    one trick when a match occurs. Diagonal (same-pattern) is invalid (-1).

    Returns:
    scores : np.ndarray
        Matrix where scores[i, j] is P1's trick count when P1 uses pattern i
        against P2 pattern j.
    ties : np.ndarray, optional
        If ``return_ties`` is True, also return an 8x8 matrix flagging ties
        (1 for tie, 0 for decisive result, -1 on the diagonal).
    """
    assert deck.shape == (52,)
    assert np.sum(deck) == 26

    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    scores = np.full((8, 8), -1, dtype=np.int16)
    tie_flags = np.full((8, 8), -1, dtype=np.int16) if return_ties else None

    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i == j:
                continue
            p1c, p2c = _count_both(deck, p1, p2)
            scores[i, j] = p1c
            if return_ties:
                tie_flags[i, j] = int(p1c == p2c)

    if return_ties:
        return scores, tie_flags
    return scores


def score_humble_nishiyama_cards(deck: np.ndarray,*,
                                 return_ties: bool = False,) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Variant of the Humble–Nishiyama score that tracks total cards collected.

    Whenever a player's pattern appears, they claim every card flipped since the
    previous win (inclusive). The diagonal remains invalid (-1).
    """
    assert deck.shape == (52,)
    assert np.sum(deck) == 26

    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    scores = np.full((8, 8), -1, dtype=np.int16)
    tie_flags = np.full((8, 8), -1, dtype=np.int16) if return_ties else None

    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i == j:
                continue
            p1_cards, p2_cards = _count_cards(deck, p1, p2)
            scores[i, j] = p1_cards
            if return_ties:
                tie_flags[i, j] = int(p1_cards == p2_cards)

    if return_ties:
        return scores, tie_flags
    return scores


def p2_win_prob_matrix(n_games: int = 100, base_seed: int = 2024) -> np.ndarray:
    """
    Estimate P2's win probability per matchup (P1 pattern i, P2 pattern j)
    across `n_games` random decks. A win is counted when P2's total tricks > P1's.

    Returns an 8x8 float array with diagonal set to NaN (invalid same-pattern).
    """
    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    decks = get_decks(n_games, seed=base_seed)

    win_sum = np.zeros((8, 8), dtype=int)
    game_cnt = np.zeros((8, 8), dtype=int)

    for deck in decks:
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


def p2_win_prob_from_mats(
    mats: np.ndarray, *,return_ties = True,) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute P2 win probabilities from a batch of Humble–Nishiyama matrices.
    `mats` shape should be (n_decks, 8, 8), where mats[d, i, j] is P1's score
    on deck d when P1 uses pattern i and P2 uses pattern j.

    Returns an 8x8 float array with diagonal set to NaN, where entry [i, j]
    is the fraction of decks for which P2's score > P1's score when P1 picks i
    and P2 picks j. P2's score on a deck for (i, j) equals mats[d, j, i].

    If ``return_ties`` is True, also returns an 8x8 array of tie frequencies.
    """
    if mats.ndim != 3 or mats.shape[1:] != (8, 8):
        raise ValueError("mats must have shape (n, 8, 8)")
    n = mats.shape[0]
    flipped = mats[:, :, :].transpose(0, 2, 1)
    p2_wins = (flipped > mats).sum(axis=0)
    win_probs = p2_wins / float(n)
    win_probs[np.eye(8, dtype=bool)] = np.nan

    if not return_ties:
        return win_probs

    tie_counts = (flipped == mats).sum(axis=0)
    tie_probs = tie_counts / float(n)
    tie_probs[np.eye(8, dtype=bool)] = np.nan
    return win_probs, tie_probs
