# Read Me

### Highlights

* **Game Foundation:** Two players select one of eight possible 3-card binary patterns, and the winner is the player whose pattern appears most frequently in a 52-card deck.
* **The Penney's Game Bias:** The card game fundamentally originates in the Penney's Game; for any pattern P1 chooses, P2 can always choose a different pattern that has a more favorable probability of winning. 
* **The Simulated Scoring Variants:**
    * **Trick-Counting:** Players win a 'trick' every time their pattern appears in a non-overlapping manner. Simulations confirm P2 maintains the advantage, meaning P2 can always find a winning counter-strategy.
    * **Card-Counting:** When a players pattern matches the sequence, they win all the cards since the last pattern match.
* **Winning the Variants:** The advantage for P2 is minimally weakened by the card-counting variant according the the simulation, yet P2 still experiences the bias. 

## Card Game Environment

This repository simulates and analyzes a card game based on the Humble–Nishiyama Game with focus on how different scoring methods impact the optimal strategy for P2. The simulation utilizes two players each selecting a pattern represented as 3-bit binary strings. The players try to match the patterns in a shuffled 52-card deck and collect points through matches. Through five million deck simulations, the project compares the original trick-counting method with a card-counting variant, demonstrating a slight difference.

### Overview
#### 1. Trick-Counting

In this variant, the deck is scanned for patterns, and the first match wins a 'trick'. The pattern match scans three cards in advance to avoid pattern overlap, and the player with the most tricks at the end wins the game.

* **P2 Strategy:** Consistent with the original nature of the Humble–Nishiyama Game, the simulation shows that P2 can always identify a pattern that yields a winning probability against any of P1's eight choices. The resulting 'By Tricks' heatmap clearly illustrates this strategic landscape.

#### 2. Card-Counting 

In this variant, the pot of cards grows with each card revealed. The player whose pattern matches is awarded all the cards currently in the pot, and the pot resets to zero. The player with the highest total of cards at the end wins.

* **P2 Strategy:** The shift from winning a fixed trick to winning a variable pot of cards alters the optimal strategy. The simulation demonstrates that the best choice for P2 in the trick-counting game is not necessarily the best in the card-counting game. The 'By Cards' heatmap highlights this strategic difference and reveals different winning probabilities with a change in one optimal choice.

## Quick Start Guide:

The Card Game Project is managed using [UV](https://docs.astral.sh/uv/guides/install-python/). To check if UV is installed run uv in terminal. If successfully installed, the uv help and command menu will appear. Otherwise, follow the install guidelines on [UV's website](https://docs.astral.sh/uv/guides/install-python/).
<br><br>
With UV properly installed, download this repository and once navigated to the directory, run uv sync to install Card_Game dependencies. 
<br><br>
To run Card Game: uv run main.py
<br><br>
Running the program will score any un-scored decks and then start the user interface where the user may choose to add new decks to the running total. The user may also give a specific seed for the new decks or skip, defaulting to a random seed. 

## Contents 

`main.py`: The entry point into the Card Game project. Contains N_DECKS (the number of decks that will/are generated. Defaults to 5,000,000), BATCH_SIZE (the size in which decks are scored. Defaults to 100,000), and BASE_SEED (the seed in which N_DECKS is generated from. Defaults to 2003). These values can all be edited within the file but if the user wants to increase the number of decks, they can also do so by running main.py and opting to add x more decks with y (optional) base seed. If the user opts not to add more decks, the program will return the heatmaps for the score by cards and score by tricks versions of the Humble-Nishiyama game. 

`src/`: The source code file that contains gen_data.py, score_data.py, viz_data.py, and utils.py. gen_data.py contains the function needed to generate the 52 card decks and a helper function to load saved decks. score_data.py contains all the functions needed to score the decks and also format said scores so that they can be plotted on the heatmap. viz_data.py contains the general plotting function for the heatmaps. utils.py contains a decorator function that tracks run time and file sizes and was used during testing. 

`data/`: The data folder which contains the raw 5,000,000 million decks saved in batches as .npy files, the manual_decks_scored.npy file of additionally user added decks, and the score_summary.npy file which contains all scores for the 5,000,000 + x amount of user generated decks. 

`figures/`: The folder in which the two heatmaps are stored. Note that each time the program is run, the figures are re-generated and replace the current two figures in the folder. 