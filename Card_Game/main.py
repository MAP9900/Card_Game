#Imports
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
from joblib import Parallel, delayed
from src.gen_data import get_decks
from src.score_data import score_humble_nishiyama, score_humble_nishiyama_cards
from src.viz_data import save_p2_win_prob_heatmap_from_counts



N_DECKS = 5_000_000
BATCH_SIZE = 100_000 #The bacth size in which decks are saved and scored. 
BASE_SEED = 2003
DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUMMARY_FILE = DATA_DIR / "score_summary.npz"
MANUAL_SAVE_FILE = DATA_DIR / "manual_decks_scored.npy"


PATTERN_COUNT = 8
DIAG_MASK = np.eye(PATTERN_COUNT, dtype=bool)
PARALLEL_CHUNK_SIZE = 512


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _empty_counts() -> dict[str, np.ndarray]:
    zero = np.zeros((PATTERN_COUNT, PATTERN_COUNT), dtype=np.int64)
    return {
        "p2_trick_wins": zero.copy(),
        "trick_ties": zero.copy(),
        "p2_card_wins": zero.copy(),
        "card_ties": zero.copy(),}


def _save_summary(total_decks: int, counts: dict[str, np.ndarray]) -> None:
    np.savez(
        SUMMARY_FILE,
        total_decks=np.array(total_decks, dtype=np.int64),
        p2_trick_wins=counts["p2_trick_wins"],
        trick_ties=counts["trick_ties"],
        p2_card_wins=counts["p2_card_wins"],
        card_ties=counts["card_ties"],)

#Loads existing scored decks
def _load_summary() -> tuple[int, dict[str, np.ndarray]]:
    if not SUMMARY_FILE.exists():
        return 0, _empty_counts()

    with np.load(SUMMARY_FILE) as data:
        total = int(data["total_decks"])
        counts = {
            "p2_trick_wins": np.array(data["p2_trick_wins"], dtype=np.int64),
            "trick_ties": np.array(data["trick_ties"], dtype=np.int64),
            "p2_card_wins": np.array(data["p2_card_wins"], dtype=np.int64),
            "card_ties": np.array(data["card_ties"], dtype=np.int64),}
    return total, counts

#Helper Function to print batch number lines when scoring
def _next_auto_batch_index() -> int:
    max_idx = -1
    prefix = "decks_auto_batch"
    for path in DATA_DIR.glob(f"{prefix}[0-9]*.npy"):
        stem = path.stem
        suffix = stem.replace(prefix, "", 1)
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1

#User Input Fucntion:
def augment_data() -> np.ndarray:
    try:
        raw_count = input("How many additional decks would you like to add? (0 to skip): ").strip()
    except EOFError:
        return np.empty((0, 52), dtype=np.uint8)
    if not raw_count:
        return np.empty((0, 52), dtype=np.uint8)
    try:
        count = int(raw_count)
    except ValueError:
        print("Invalid number entered; skipping deck generation.")
        return np.empty((0, 52), dtype=np.uint8)
    if count <= 0:
        return np.empty((0, 52), dtype=np.uint8)
    try:
        raw_seed = input("Optional: enter a seed for reproducibility (blank for random): ").strip()
    except EOFError:
        raw_seed = ""
    if raw_seed:
        try:
            seed = int(raw_seed)
        except ValueError:
            print("Seed must be an integer. Using random seed instead.")
            seed = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max))
    else:
        seed = int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max))
    decks = get_decks(count, seed=seed)
    stacked = np.ascontiguousarray(decks, dtype=np.uint8)
    np.save(MANUAL_SAVE_FILE, stacked)
    print(f"Generated {stacked.shape[0]} deck(s) using seed {seed}.")
    return stacked

def _score_batch(decks: np.ndarray, counts: dict[str, np.ndarray]) -> None:
    if decks.size == 0:
        return

    def _score_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local_trick_wins = np.zeros((PATTERN_COUNT, PATTERN_COUNT), dtype=np.int64)
        local_trick_ties = np.zeros_like(local_trick_wins)
        local_card_wins = np.zeros_like(local_trick_wins)
        local_card_ties = np.zeros_like(local_trick_wins)

        for deck in chunk:
            trick_scores = score_humble_nishiyama(deck)
            card_scores = score_humble_nishiyama_cards(deck)

            trick_wins = (trick_scores.T > trick_scores).astype(np.int64)
            trick_ties = (trick_scores.T == trick_scores).astype(np.int64)
            card_wins = (card_scores.T > card_scores).astype(np.int64)
            card_ties = (card_scores.T == card_scores).astype(np.int64)

            trick_wins[DIAG_MASK] = 0
            trick_ties[DIAG_MASK] = 0
            card_wins[DIAG_MASK] = 0
            card_ties[DIAG_MASK] = 0

            local_trick_wins += trick_wins
            local_trick_ties += trick_ties
            local_card_wins += card_wins
            local_card_ties += card_ties

        return local_trick_wins, local_trick_ties, local_card_wins, local_card_ties

    chunk_size = min(max(1, PARALLEL_CHUNK_SIZE), decks.shape[0])
    chunks = [decks[start:start + chunk_size] for start in range(0, decks.shape[0], chunk_size)]

    #Run parallel intstead of sequentially, uses all avalible cpus to speed up scoring (n_jobs=-1)
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(_score_chunk)(chunk) for chunk in chunks)

    for trick_wins, trick_ties, card_wins, card_ties in results:
        counts["p2_trick_wins"] += trick_wins
        counts["trick_ties"] += trick_ties
        counts["p2_card_wins"] += card_wins
        counts["card_ties"] += card_ties



def _score_generated_decks(counts: dict[str, np.ndarray], current_total: int) -> tuple[int, int]:
    #Avoid Rescoring the same decks
    if N_DECKS <= current_total:
        print(f"Target deck count already satisfied (total={current_total}).")
        return 0, current_total
    #Scores only needed amount of decks to get total equal to N_DECKS
    decks_needed = N_DECKS - current_total
    print(f"Generating {decks_needed} additional deck(s) to reach {N_DECKS}.")

    next_batch_index = _next_auto_batch_index()
    produced = 0
    batch_counter = 0

    while produced < decks_needed:
        batch_size = min(BATCH_SIZE, decks_needed - produced)
        seed = BASE_SEED + current_total + produced
        decks = get_decks(batch_size, seed=seed)
        batch_idx = next_batch_index + batch_counter
        np.save(DATA_DIR / f"decks_auto_batch{batch_idx:04d}.npy", decks)
        _score_batch(decks, counts)
        produced += batch_size
        batch_counter += 1
        print(f"Scored batch {batch_idx} ({batch_size} decks)")

    return produced, current_total + produced


def _build_heatmaps(total_decks: int, counts: dict[str, np.ndarray]) -> None:
    if total_decks == 0: #Safety Check
        print("No decks scored. Skipping heatmaps.")
        return

    tricks_fig = save_p2_win_prob_heatmap_from_counts(
        counts["p2_trick_wins"],
        counts["trick_ties"],
        total_decks,
        out_dir=str(FIG_DIR),
        filename="my_win_tricks.png",
        title=f"My Chance of Win(Draw)\n (By Tricks, n={total_decks})",)
    cards_fig = save_p2_win_prob_heatmap_from_counts(
        counts["p2_card_wins"],
        counts["card_ties"],
        total_decks,
        out_dir=str(FIG_DIR),
        filename="my_win_cards.png",
        title=f"My Chance of Win(Draw)\n (By Cards, n={total_decks})",)
    print("Tricks heatmap saved as:", tricks_fig)
    print("Cards heatmap saved as:", cards_fig)


def main() -> None:
    _ensure_dirs()

    current_total, counts = _load_summary()
    new_auto, updated_total = _score_generated_decks(counts, current_total)

    manual_decks = augment_data()
    if manual_decks.size:
        _score_batch(manual_decks, counts)

    total_manual = manual_decks.shape[0]
    total_decks = updated_total + total_manual

    _save_summary(total_decks, counts)
    print(f"Auto decks added: {new_auto}, manual decks: {total_manual}, total: {total_decks}")
    print(f"Saved summary to {SUMMARY_FILE}")

    _build_heatmaps(total_decks, counts)


if __name__ == "__main__":
    main()
