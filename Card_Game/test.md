# Deck Generation: Full Decks vs. Seeds (2M decks)

| Pipeline        | Generation Time | Array Size (RAM) | Disk Storage (bytes) | Disk Storage (MB) | Notes |
|-----------------|-----------------|------------------|-----------------------|-------------------|-------|
| **Decks (2M)**  | ~1.94 s         | ~832.0 MB        | 20 × 41,600,128 + 136 = **832,002,696** | ~832.0 MB (≈793.6 MiB) | 20 chunks + 1 seed file |
| **Seeds (2M)**  | ~0.02 s         | ~16.0 MB         | 16,000,128            | ~16.0 MB (≈15.3 MiB) | Single `.npy` file |

Seed method stores the 2 million seeds used to shuffle the decks. The decks only get generate when the scoring function is called but are not saved themselves, just the scores associated with the decks. The aim of this method was to save on memory by 
never saving the decks themselves but instead the seeds used to shuffle them. Decks remain reproducible as seeds used to shuffle them are saved. 


The scoring functions iterate through a 52-card deck while checking for matches against each player’s chosen 3 bit pattern. In `_count_both`, a sliding window of 3 cards is compared to player 1’s or player 2’s pattern. If a match is found, that player’s count increases and the index skips ahead by 3, otherwise it advances by 1. In `_count_cards`, the code instead tracks a “pot” of cards since the last win: once a 3-card match is found, the entire pot size is awarded to the matching player and the pot resets. The outer functions (`score_humble_nishiyama` and `score_humble_nishiyama_cards`) build an 8×8 matrix of all possible pattern matchups (skipping diagonals where both players use the same pattern), calling the counting functions to fill in results and optionally flag ties. This is done for p1 wins but by taking the inverse we can easily get p2 wins as well. 