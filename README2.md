# Read Me

uv sync <br>
uv run main.py -- Will start augment_data() which checks if the number of scored decks is less N_DECKS and will score any remaining decks. Then prompts the user to add new decks or to skip. If new decks are wanted, will score and then add to the summary and produce the updated heatmaps. 


### Highlights

* **Game Foundation:** Two players select one of eight possible 3-card binary patterns, and the winner is the player whose pattern appears most frequently in a 52-card deck.
* **The Penney's Game Bias:** The card game fundamentally originates in the Penney's Game; for any pattern P1 chooses, P2 can always choose a different pattern that has a more favorable probability of winning. 
* **The Simulated Scoring Variants:**
    * **Trick-Counting:** Players win a 'trick' every time their pattern appears in a non-overlapping manner. Simulations confirm P2 maintains the advantage, meaning P2 can always find a winning counter-strategy.
    * **Card-Counting:** When a players pattern matches the sequence, they win all the cards since the last pattern match.
* **Winning the Variants:** The advantage for P2 is minimally weakened by the card-counting variant according the the simulation, yet P2 still experiences the bias. 

## Card Game Environment

This repository simulates and analyzes a card game based on the Humble–Nishiyama Game with focus on how different scoring methods impact the optimal strategy for P2. The simulation utilizes two players each selecting a pattern represented as 3-bit binary strings. The players try to match the patterns in a shuffled 52-card deck and collect points through matches. Through five million deck simulations, the project compares the original trick-counting method with a card-counting variant, demonstrating a slight difference.

## Project Structure and Usage
The repository consists of the following Python files:

main.py: The entry point for the simulation. It orchestrates the entire process: loading existing data, generating new random decks to meet a target count (N_DECKS = 5,000,000), scoring the decks in parallel, saving the cumulative win/tie counts, and generating the final win-probability heatmaps for both scoring variants.

gen_data.py: Handles the efficient generation of 52-card decks (26 '0's and 26 '1's) and manages random seeds for reproducibility.

score_data.py: Contains the core logic, including highly optimized numba-compiled functions (_score_tricks, _score_cards) to quickly determine trick or card counts for all 56 unique P1 vs. P2 pattern matchups on a single deck. It also calculates the P2 win probability matrices from the aggregated scores.

viz_data.py: Contains functions for creating and saving the strategy heatmaps. It calculates the win and tie percentages for P2 against P1 for every pattern pair and visually highlights the best choice for P2 in each row (i.e., P2's best counter-strategy against P1's pattern).

### Overview
#### 1. Trick-Counting

In this variant, the deck is scanned for patterns, and the first match wins a 'trick'. The pattern match scans three cards in advance to avoid pattern overlap, and the player with the most tricks at the end wins the game.

* **P2 Strategy:** Consistent with the original nature of the Humble–Nishiyama Game, the simulation shows that P2 can always identify a pattern that yields a winning probability against any of P1's eight choices. The resulting 'By Tricks' heatmap clearly illustrates this strategic landscape.

#### 2. Card-Counting 

In this variant, the pot of cards grows with each card revealed. The player whose pattern matches is awarded all the cards currently in the pot, and the pot resets to zero. The player with the highest total of cards at the end wins.

* **P2 Strategy:** The shift from winning a fixed trick to winning a variable pot of cards alters the optimal strategy. The simulation demonstrates that the best choice for P2 in the trick-counting game is not necessarily the best in the card-counting game. The 'By Cards' heatmap highlights this strategic difference and reveals different winning probabilities with a change in one optimal choice.

The project's code base is structured to handle the simulation: `gen_data.py` generates the decks, `score_data.py` uses fast `numba` kernels to score the outcomes, and `main.py` utilizes parallel processing with `joblib` to aggregate the results for five million decks before `viz_data.py` generates the final heatmaps.
