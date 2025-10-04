import itertools
import os
import csv
import numpy as np
from src.gen_data import get_decks, load_decks


#Way one to score decks 

# -------- Penney's game (tricks) --------
def score_tricks(deck, p1_seq, p2_seq):
    """Return winner: 1, 2, or 0 (draw) based on sequence counts."""
    p1_count, p2_count = 0, 0

    for i in range(len(deck) - 2):
        window = tuple(deck[i:i+3])
        if window == tuple(p1_seq):
            p1_count += 1
        elif window == tuple(p2_seq):
            p2_count += 1

    if p1_count > p2_count:
        return 1
    elif p2_count > p1_count:
        return 2
    else:
        return 0


# -------- Humble–Nishiyama game (cards) --------
def score_cards(deck, p1_seq, p2_seq):
    """Return winner: 1, 2, or 0 (draw) for card rules."""
    pot = []
    p1_cards, p2_cards = 0, 0

    for i, card in enumerate(deck):
        pot.append(card)
        if i >= 2:
            window = tuple(pot[-3:])
            if window == tuple(p1_seq):
                p1_cards += len(pot)
                pot = []
            elif window == tuple(p2_seq):
                p2_cards += len(pot)
                pot = []

    # compare totals
    if p1_cards > p2_cards:
        return 1
    elif p2_cards > p1_cards:
        return 2
    else:
        return 0


# -------- Simulation --------
def simulate_matchup(decks, p1_seq, p2_seq):
    """Simulate Penney’s and Humble–Nishiyama for one matchup."""
    results = {
        "Tricks_P1_Wins": 0, "Tricks_P2_Wins": 0, "Tricks_Draws": 0,
        "Cards_P1_Wins": 0, "Cards_P2_Wins": 0, "Cards_Draws": 0,
    }

    for deck in decks:
        # Penney’s
        t_winner = score_tricks(deck, p1_seq, p2_seq)
        if t_winner == 1:
            results["Tricks_P1_Wins"] += 1
        elif t_winner == 2:
            results["Tricks_P2_Wins"] += 1
        else:
            results["Tricks_Draws"] += 1

        # Humble–Nishiyama
        c_winner = score_cards(deck, p1_seq, p2_seq)
        if c_winner == 1:
            results["Cards_P1_Wins"] += 1
        elif c_winner == 2:
            results["Cards_P2_Wins"] += 1
        else:
            results["Cards_Draws"] += 1

    return results


# -------- Main --------
def main():
    # Load existing decks (change filename to match your saved batch)
    PATH_TO_DECK = os.path.join("results", "decks_0.npy")
    decks, seed = load_decks(filename=os.path.basename(PATH_TO_DECK))

    print(f"Loaded {len(decks)} decks with seed {seed}")

    # All 8 sequences of 3 cards
    seqs = [tuple(map(int, f"{i:03b}")) for i in range(8)]

    results = []

    for p1_seq, p2_seq in itertools.permutations(seqs, 2):
        matchup = simulate_matchup(decks, p1_seq, p2_seq)
        results.append({
            "Player1": "".join(map(str, p1_seq)),
            "Player2": "".join(map(str, p2_seq)),
            **matchup
        })

    # Save CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "results/game_simulation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {csv_path}")



if __name__ == "__main__":
    main()
