import numpy as np
import os
from src.gen_data import generate_seeds, compute_scores_from_seeds, save_seeds, load_seeds, save_scores,load_scores
from src.score_data import score_humble_nishiyama
from src.viz_data import save_hn_score_heatmap, save_p2_win_prob_heatmap_from_mats

# def simple_score(deck: np.ndarray) -> int:
#     return int(deck[:10].sum())

# def main():
#     seeds = generate_seeds(5, base_seed=1976)
#     scores = compute_scores_from_seeds(seeds, simple_score)
#     print("seeds:", seeds)
#     print("scores:", scores)


if __name__ == "__main__":  
    #Test of 2 mil seeds
    n = 50_000
    seeds_file = f"seeds_{n}.npy"
    scores_file = f"scores_{n}.npy"

    #Seeds
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, seeds_file)):
        seeds_big = load_seeds(seeds_file)
        print(f"Loaded seeds from data/{seeds_file}")
    else:
        seeds_big = generate_seeds(n, base_seed=2024)
     
        save_seeds(seeds_big, seeds_file)

    # Scores (per-deck 8x8 matrices)
    if os.path.exists(os.path.join(data_dir, scores_file)):
        mats = load_scores(scores_file)
        print(f"Loaded scores from data/{scores_file}")
    else:
        mats = compute_scores_from_seeds(seeds_big, score_humble_nishiyama)
        save_scores(mats, scores_file)

    # out3 = save_p2_win_prob_heatmap_from_mats(mats).  #Not needed for testing
    # print(f"Saved combined probability heatmap to: {out3}")


#MAP Test results 1 (Canceled Scoring as estimation)

# (base) matthewplambeck@Matthews-MacBook-Pro-2 Card_Game % uv run python -m src.test

# [time_and_size] save_seeds elapsed: 9.36 ms
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/seeds_2000000.npy (16000128 bytes)
# ^CTraceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code
#   File "/Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/src/test.py", line 39, in <module>
#     mats = compute_scores_from_seeds(seeds_big, score_humble_nishiyama)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/src/gen_data.py", line 53, in compute_scores_from_seeds
#     out[i] = np.asarray(score_fn(deck_from_seed(int(seeds[i]))))
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/src/score_data.py", line 48, in score_humble_nishiyama
#     if np.array_equal(window, p1):
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/.venv/lib/python3.12/site-packages/numpy/_core/numeric.py", line 2524, in array_equal
#     @array_function_dispatch(_array_equal_dispatcher)
    
# KeyboardInterrupt
# (base) matthewplambeck@Matthews-MacBook-Pro-2 Card_Game % 

#Test 2, n = 50,000

# (base) matthewplambeck@Matthews-MacBook-Pro-2 Card_Game % uv run python -m src.test

# [time_and_size] save_seeds elapsed: 0.39 ms
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/seeds_50000.npy (400128 bytes)
# [time_and_size] save_scores elapsed: 7.52 ms.        Something is wrong with the timing function, took way longer than 7.52 ms. Likely need the decorator above the actual scoring for loop funciton
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/scores_50000.npy (6400128 bytes)









#To Do: Clean up test.py/create run_tests.py, turn gen_data into a class, understand/clean up score_data, 


