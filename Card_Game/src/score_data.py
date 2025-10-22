import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Tuple
import numpy as np
from numba import njit
from src.gen_data import get_decks



#numpy constants reused by the JIT compiled scoring kernels
PATTERNS = np.array([[(i >> (2 - bit)) & 1 for bit in range(3)] for i in range(8)], dtype=np.uint8,)


@njit(cache=True)
def _score_tricks(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    p1c = 0
    p2c = 0
    idx = 0
    n = deck.shape[0]
    while idx <= n - 3:
        match_p1 = True
        for k in range(3):
            if deck[idx + k] != p1[k]:
                match_p1 = False
                break
        if match_p1:
            p1c += 1
            idx += 3
            continue

        match_p2 = True
        for k in range(3):
            if deck[idx + k] != p2[k]:
                match_p2 = False
                break
        if match_p2:
            p2c += 1
            idx += 3
        else:
            idx += 1
    return p1c, p2c


@njit(cache=True)
def _score_cards(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    p1_cards = 0
    p2_cards = 0
    pot_start = 0
    idx = 0
    n = deck.shape[0]
    while idx < n:
        pot_size = idx - pot_start + 1
        if pot_size >= 3:
            match_p1 = True
            for k in range(3):
                if deck[idx - 2 + k] != p1[k]:
                    match_p1 = False
                    break
            if match_p1:
                p1_cards += pot_size
                pot_start = idx + 1
            else:
                match_p2 = True
                for k in range(3):
                    if deck[idx - 2 + k] != p2[k]:
                        match_p2 = False
                        break
                if match_p2:
                    p2_cards += pot_size
                    pot_start = idx + 1
        idx += 1
    return p1_cards, p2_cards


@njit(cache=True)
def _score_humble_nishiyama(deck: np.ndarray, return_ties: bool) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full((8, 8), -1, dtype=np.int16)
    tie_flags = np.full((8, 8), -1, dtype=np.int16)
    for i in range(8):
        p1 = PATTERNS[i]
        for j in range(8):
            if i == j:
                continue
            p2 = PATTERNS[j]
            p1c, p2c = _score_tricks(deck, p1, p2)
            scores[i, j] = p1c
            tie_flags[i, j] = 1 if p1c == p2c else 0
    if return_ties:
        return scores, tie_flags
    return scores, tie_flags


@njit(cache=True)
def _score_humble_nishiyama_cards(deck: np.ndarray, return_ties: bool) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full((8, 8), -1, dtype=np.int16)
    tie_flags = np.full((8, 8), -1, dtype=np.int16)
    for i in range(8):
        p1 = PATTERNS[i]
        for j in range(8):
            if i == j:
                continue
            p2 = PATTERNS[j]
            p1_cards, p2_cards = _score_cards(deck, p1, p2)
            scores[i, j] = p1_cards
            tie_flags[i, j] = 1 if p1_cards == p2_cards else 0
    if return_ties:
        return scores, tie_flags
    return scores, tie_flags

def _ensure_deck(deck: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(deck, dtype=np.uint8)
    if arr.shape != (52,):
        raise ValueError("Deck must be a 1D array of length 52.")
    if int(arr.sum()) != 26:
        raise ValueError("Deck must contain exactly 26 ones (and 26 zeros).")
    return arr


def _ensure_pattern(pattern: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(pattern, dtype=np.uint8)
    if arr.shape != (3,):
        raise ValueError("Patterns must be 3-card sequences.")
    return arr

def score_humble_nishiyama(deck: np.ndarray, *, return_ties: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Scores the Humble–Nishiyama trick counts for all pattern matchups.
    Returns the P1 score matrix, and optionally tie flags when ``return_ties`` is True.
    """
    deck_arr = _ensure_deck(deck)
    scores, ties = _score_humble_nishiyama(deck_arr, return_ties)
    if return_ties:
        return scores, ties
    return scores

def score_humble_nishiyama_cards(deck: np.ndarray, *, 
                                 return_ties: bool = False,) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Variant of Humble–Nishiyama scoring that tallies total cards collected.
    Returns the P1 score matrix, and optionally tie flags when ``return_ties`` is True.
    """
    deck_arr = _ensure_deck(deck)
    scores, ties = _score_humble_nishiyama_cards(deck_arr, return_ties)
    if return_ties:
        return scores, ties
    return scores


def score_tricks(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    """
    Count how many tricks P1 and P2 take on a single deck given their 3-bit patterns.
    Returns (p1_count, p2_count).
    """
    deck_arr = _ensure_deck(deck)
    p1_arr = _ensure_pattern(p1)
    p2_arr = _ensure_pattern(p2)
    p1c, p2c = _score_tricks(deck_arr, p1_arr, p2_arr)
    return int(p1c), int(p2c)


def score_cards(deck: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
    """
    Return card totals awarded to each player using the pot scoring variant.
    Returns (p1_count, p2_count).
    """
    deck_arr = _ensure_deck(deck)
    p1_arr = _ensure_pattern(p1)
    p2_arr = _ensure_pattern(p2)
    p1_cards, p2_cards = _score_cards(deck_arr, p1_arr, p2_arr)
    return int(p1_cards), int(p2_cards)



def p2_win_prob_matrix(n_games: int = 100, base_seed: int = 2024) -> np.ndarray:
    """
    Estimate P2's win probability per matchup (P1 pattern i, P2 pattern j)
    across `n_games` random decks. A win is counted when P2's total tricks > P1's.

    Returns an 8x8 float array with diagonal set to NaN (invalid same-pattern).
    """
    decks = get_decks(n_games, seed=base_seed)

    win_sum = np.zeros((8, 8), dtype=int)
    game_cnt = np.zeros((8, 8), dtype=int)

    for deck in decks:
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue #Skip diagonal 
                p1c, p2c = score_tricks(deck, PATTERNS[i], PATTERNS[j])
                game_cnt[i, j] += 1
                if p2c > p1c:
                    win_sum[i, j] += 1

    with np.errstate(invalid='ignore'):
        probs = win_sum / np.maximum(game_cnt, 1)
    probs[np.eye(8, dtype=bool)] = np.nan
    return probs


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

