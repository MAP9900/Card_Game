import numpy as np

def generate_seeds(n: int, base_seed = 12345) -> np.ndarray:
    """
    Return n independent uint64 seeds based off set base seed
    """
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)

def deck_from_seed(seed: int) -> np.ndarray:
    """
    Return a 52-card deck with 26 zeros (B) and 26 ones (R), shuffled by generated seeds
    """
    rng = np.random.default_rng(seed)
    deck = np.empty(52, dtype=np.uint8)
    deck[:26] = 0
    deck[26:] = 1
    rng.shuffle(deck)
    return deck


def compute_scores_from_seeds(seeds: np.ndarray, score_fn) -> np.ndarray:
    """
    Rebuild each deck from seed and score it with `score_fn(deck)`.
    Supports score functions that return either:
      - a scalar (int/float), or
      - a NumPy array of any shape
         
    Returns an array shaped as:
      - (len(seeds),)                 for scalar scores
      - (len(seeds), *score_shape)    for array scores
    """
    n = len(seeds)
    if n == 0:
        return np.array([], dtype=np.int16)

    #Probe first result
    first = score_fn(deck_from_seed(int(seeds[0])))
    if np.isscalar(first):
        out = np.empty(n, dtype=np.asarray(first).dtype)
        out[0] = first
        for i in range(1, n):
            out[i] = score_fn(deck_from_seed(int(seeds[i])))
        return out
    else:
        first_arr = np.asarray(first)
        out = np.empty((n, *first_arr.shape), dtype=first_arr.dtype)
        out[0] = first_arr
        for i in range(1, n):
            out[i] = np.asarray(score_fn(deck_from_seed(int(seeds[i]))))
        return out

def score_humble_nishiyama(deck: np.ndarray) -> np.ndarray:
    """
    Scores 8x8 matrix of all valid player matchups (P1 != P2 only).
    Each player picks a 3-card pattern (000–111), and the game is
    played by flipping the deck and scoring when a match occurs.
    
    Returns:
        scores[i, j] = Player 1's score when P1 picks pattern i, P2 picks j.
                      Diagonal values (i == j) are set to -1 (invalid).
    """
    assert deck.shape == (52,)
    assert np.sum(deck) == 26

    patterns = [np.array([int(b) for b in f"{i:03b}"]) for i in range(8)]
    scores = np.full((8, 8), -1, dtype=np.int16)  # default all to invalid (-1)

    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i == j:
                continue  #skip same-pattern matchups
            score = 0
            idx = 0
            while idx <= 49:  #last window: deck[49:52]
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
