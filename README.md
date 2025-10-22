# Read Me

## Purpose

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

