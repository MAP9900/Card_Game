#Imports
import os
import numpy as np
from src.gen_data import get_decks, save_decks
from src.score_data import (
    export_hn_scores_to_csv,
    score_humble_nishiyama,
    score_humble_nishiyama_cards,)
from src.viz_data import save_p2_win_prob_heatmap_from_mats


#Runs tests for generating decks and scoring from score_data.py. Also saves scores to csv and creates test heatmaps. 

def _data_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    d = os.path.join(base_dir, "data")
    os.makedirs(d, exist_ok=True)
    return d


def _save_scores(array: np.ndarray, filename: str) -> str:
    path = os.path.join(_data_dir(), filename)
    np.save(path, array)
    return path


def main() -> None:
    n_decks = 150 #Can be changed
    seed = 2024

    decks = get_decks(n_decks, seed)
    deck_files = save_decks(decks, seed, filename="decks_test.npy")

    mats_tricks = np.stack([score_humble_nishiyama(deck) for deck in decks])
    mats_cards = np.stack([score_humble_nishiyama_cards(deck) for deck in decks])

    tricks_path = _save_scores(mats_tricks, "scores_tricks_150.npy")
    cards_path = _save_scores(mats_cards, "scores_cards_150.npy")
    csv_path = export_hn_scores_to_csv(
        tricks_path,
        cards_scores_file=cards_path,
        out_csv=os.path.join(_data_dir(), "scores_combined_150.csv"),
    )

    tricks_fig = save_p2_win_prob_heatmap_from_mats(
        mats_tricks, filename="p2_win_tricks_test.png"
    )
    cards_fig = save_p2_win_prob_heatmap_from_mats(
        mats_cards, filename="p2_win_cards_test.png"
    )

    print(f"Saved decks to: {deck_files}")
    print(f"Saved trick-count matrices to: {tricks_path}")
    print(f"Saved card-count matrices to: {cards_path}")
    print(f"Saved combined CSV to: {csv_path}")
    print(f"Saved P2 win heatmap (tricks) to: {tricks_fig}")
    print(f"Saved P2 win heatmap (cards) to: {cards_fig}")


if __name__ == "__main__":
    main()
