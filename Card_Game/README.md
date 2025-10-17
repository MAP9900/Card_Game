**To run the card game simulator:**

uv sync <br>
uv run main.py -- Will start augment_data() which checks if the number of scored decks is less N_DECKS and will score any remaining decks. Then prompts the user to add new decks or to skip. If new decks are wanted, will score and then add to the summary and produce the updated heatmaps. 