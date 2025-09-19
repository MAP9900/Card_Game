# Deck Generation: Full Decks vs. Seeds (2M decks)

| Pipeline        | Generation Time | Array Size (RAM) | Disk Storage (bytes) | Disk Storage (MB) | Notes |
|-----------------|-----------------|------------------|-----------------------|-------------------|-------|
| **Decks (2M)**  | ~1.94 s         | ~832.0 MB        | 20 × 41,600,128 + 136 = **832,002,696** | ~832.0 MB (≈793.6 MiB) | 20 chunks + 1 seed file |
| **Seeds (2M)**  | ~0.02 s         | ~16.0 MB         | 16,000,128            | ~16.0 MB (≈15.3 MiB) | Single `.npy` file |


Both of us chose different methods for storing data; one focusing on seed-based deck reconstruction for immediate scoring, and the other method pursuing bulk deck generation. While both methods accomplish reproducibility and generation of two million decks available for the Penney's and Humble-Nishiyama Randomness games, basic benchmark results reveal a general preference towards seed-based regeneration. However, we have a personal preference for the familiarity of the bulk deck generation. 

The direct bulk storage uses twenty batches of 100,000 decks to avoid GitHub file constraints alongside a seed file for reproducibility. The process involved generating a binary array (2,000,000, 52) with a random number generator and permutation to encourage independent shuffling. After array generation, the twenty batches are saved in a separate data file for eventual access by the games in main.py. The seed-based method stores the 2 million seeds used to shuffle the decks. The decks only get generated when the scoring function is called but are not saved themselves, just the scores associated with the decks. The aim of this method was to save on memory by never saving the decks themselves but instead the seeds used to shuffle them. Decks remain reproducible as seeds used to shuffle them are saved. 

We compared our results using the same tests with the assistance of different debuggers, then compared the results with a shared debugger to compare capabilities. After multiple runs, the memory usage and average runtime favors the seed-based pipeline. Revealed as the most notable difference through the benchmark results, bulk deck generation consumes over fifty times the memory usage than the seed-based decks. Alongside memory usage, the overall seed-based generation runs four times faster than the bulk generation; however, the read time is nominally smaller for the bulk method. 

Regardless of the various metrics in favor of the seed-based method, since the bulk deck generation additionally saves the seeds, the method will prove slightly simpler. Through game simulation, we discovered issues with the seed-based method is less intuitive. While we haven't accomplished the scoring aspect completely, the bulk generation better represents the independent nature of the card flip. For the purposes of playing Penney's and H-R Randomness games, we believe the memory usage cost will prove beneficial for simplicity. 

Final Note: This repo has two src files, one for each method. Both have their own respective run_tests.py files and will run independently of each other. 