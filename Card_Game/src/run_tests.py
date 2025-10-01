import src
from src.gen_data import (
    compute_scores_from_seeds,
    generate_seeds,
    save_scores,
    save_seeds,
)
from src.score_data import score_humble_nishiyama, score_humble_nishiyama_cards
from src.viz_data import save_p2_win_prob_heatmap_from_mats


def main() -> None:
    n = 100
    base_seed = 2024

    seeds = generate_seeds(n, base_seed=base_seed)
    mats_tricks = compute_scores_from_seeds(seeds, score_humble_nishiyama)
    mats_cards = compute_scores_from_seeds(seeds, score_humble_nishiyama_cards)

    seeds_path = save_seeds(seeds, "seeds_test_100.npy")
    tricks_path = save_scores(mats_tricks, "scores_tricks_100.npy")
    cards_path = save_scores(mats_cards, "scores_cards_100.npy")

    tricks_fig = save_p2_win_prob_heatmap_from_mats(
        mats_tricks,
        filename="p2_win_tricks_test.png",
    )
    cards_fig = save_p2_win_prob_heatmap_from_mats(
        mats_cards,
        filename="p2_win_cards_test.png",
    )

    print(f"Saved seeds to: {seeds_path}")
    print(f"Saved trick-count matrices to: {tricks_path}")
    print(f"Saved card-count matrices to: {cards_path}")
    print(f"Saved P2 win heatmap (tricks) to: {tricks_fig}")
    print(f"Saved P2 win heatmap (cards) to: {cards_fig}")


if __name__ == "__main__":
    main()
