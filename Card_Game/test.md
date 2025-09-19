# Deck Generation: Full Decks vs. Seeds (2M decks)

| Pipeline        | Generation Time | Array Size (RAM) | Disk Storage (bytes) | Disk Storage (MB) | Notes |
|-----------------|-----------------|------------------|-----------------------|-------------------|-------|
| **Decks (2M)**  | ~1.94 s         | ~832.0 MB        | 20 × 41,600,128 + 136 = **832,002,696** | ~832.0 MB (≈793.6 MiB) | 20 chunks + 1 seed file |
| **Seeds (2M)**  | ~0.02 s         | ~16.0 MB         | 16,000,128            | ~16.0 MB (≈15.3 MiB) | Single `.npy` file |

Seed method stores the 2 million seeds used to shuffle the decks. The decks only get generate when the scoring function is called but are not saved themselves, just the scores associated with the decks. The aim of this method was to save on memory by 
never saving the decks themselves but instead the seeds used to shuffle them. Decks remain reproducible as seeds used to shuffle them are saved. 