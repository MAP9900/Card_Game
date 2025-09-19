import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Our existing modules
from helpers import PATH_DATA
from datagen import get_decks, save_decks, load_decks
from gen_data import (
    generate_seeds,
    deck_from_seed,
    compute_scores_from_seeds,
    save_seeds,
    save_scores,
)

#TO RUN: uv run python Card_Game/src_EP/run_tests.py 


# ----------------------------
# Benchmark utilities
# ----------------------------
def benchmark(func, *args, repeat: int = 5, **kwargs):
    """Run function multiple times and return mean, std of runtime (seconds)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def memory_usage_bytes(arr: np.ndarray) -> int:
    """Return memory usage of NumPy array in bytes."""
    return arr.nbytes


# ----------------------------
# Run tests
# ----------------------------
RUN_BENCHMARKS = False
RUN_DECK_PIPELINE = True
RUN_SEED_PIPELINE = True
RUN_SCORING = False  # Computing scores for 2M decks is very slow; enable intentionally.


def run_tests(n_decks: int = 2_000_000, seed: int = 12345, batch_size: int = 100_000):
    """
    Run reproducibility + performance tests for both datagen and seed pipelines.
    """
    os.makedirs(PATH_DATA, exist_ok=True)

    if RUN_BENCHMARKS:
        results = []

        # =====================================================
        # 1. Deck pipeline (datagen.py)
        # =====================================================
        mean_gen, std_gen = benchmark(get_decks, n_decks, seed)
        sample_decks = get_decks(n_decks, seed)
        mem_bytes = memory_usage_bytes(sample_decks)

        results.append({
            "Pipeline": "Decks",
            "Test": "Deck Generation",
            "Num Decks": n_decks,
            "Batch Size": batch_size,
            "Mean Time (s)": mean_gen,
            "Std Time (s)": std_gen,
            "Memory (MB)": mem_bytes / 1e6
        })

        mean_write, std_write = benchmark(save_decks, sample_decks, seed, batch_size, "tmp_decks")
        results.append({
            "Pipeline": "Decks",
            "Test": "Disk Write",
            "Num Decks": n_decks,
            "Batch Size": batch_size,
            "Mean Time (s)": mean_write,
            "Std Time (s)": std_write,
            "Memory (MB)": mem_bytes / 1e6
        })

        batch_file = os.path.join(PATH_DATA, "tmp_decks_0.npy")
        mean_read, std_read = benchmark(np.load, batch_file)
        results.append({
            "Pipeline": "Decks",
            "Test": "Disk Read",
            "Num Decks": n_decks,
            "Batch Size": batch_size,
            "Mean Time (s)": mean_read,
            "Std Time (s)": std_read,
            "Memory (MB)": mem_bytes / 1e6
        })

        for f in Path(PATH_DATA).glob("tmp_decks*"):
            f.unlink()

        # =====================================================
        # 2. Seed pipeline (gen_data.py)
        # =====================================================
        n_seeds = n_decks  # one seed per deck

        mean_seed, std_seed = benchmark(generate_seeds, n_seeds, seed)
        seeds = generate_seeds(n_seeds, seed)
        mem_bytes_seeds = memory_usage_bytes(seeds)

        results.append({
            "Pipeline": "Seeds",
            "Test": "Seed Generation",
            "Num Decks": n_decks,
            "Batch Size": "-",
            "Mean Time (s)": mean_seed,
            "Std Time (s)": std_seed,
            "Memory (MB)": mem_bytes_seeds / 1e6
        })

        mean_rebuild, std_rebuild = benchmark(deck_from_seed, int(seeds[0]))
        results.append({
            "Pipeline": "Seeds",
            "Test": "Deck Rebuild (from seed)",
            "Num Decks": 1,
            "Batch Size": "-",
            "Mean Time (s)": mean_rebuild,
            "Std Time (s)": std_rebuild,
            "Memory (MB)": 52 / 1e6  # trivial single deck memory
        })

        # Example scoring: sum of 1s
        def example_score_fn(deck):
            return np.sum(deck)

        mean_score, std_score = benchmark(compute_scores_from_seeds, seeds[:10_000], example_score_fn)
        scores = compute_scores_from_seeds(seeds[:10_000], example_score_fn)
        mem_bytes_scores = memory_usage_bytes(scores)

        results.append({
            "Pipeline": "Seeds",
            "Test": "Compute Scores",
            "Num Decks": n_decks,
            "Batch Size": "-",
            "Mean Time (s)": mean_score,
            "Std Time (s)": std_score,
            "Memory (MB)": mem_bytes_scores / 1e6
        })

        # =====================================================
        # Save results
        # =====================================================
        df = pd.DataFrame(results)
        out_file = Path(PATH_DATA) / "benchmark_results.csv"
        df.to_csv(out_file, index=False)

        print("\nBenchmark Results:")
        print(df.to_string(index=False))
        print(f"\nResults saved to {out_file}")

    print("\n=== Direct 2M deck test ===")

    if RUN_DECK_PIPELINE:
        print("\n[Deck pipeline]")
        deck_t0 = time.perf_counter()
        decks = get_decks(n_decks, seed)
        deck_elapsed = time.perf_counter() - deck_t0
        deck_mem_mb = memory_usage_bytes(decks) / 1e6
        print(f"Generated {decks.shape[0]} decks in {deck_elapsed:.2f} s (~{deck_mem_mb:.2f} MB array)")
        save_decks(decks, seed, batch_size=batch_size, filename=f"decks_{n_decks}")
        del decks

    if RUN_SEED_PIPELINE:
        print("\n[Seed pipeline]")
        seed_t0 = time.perf_counter()
        seeds = generate_seeds(n_decks, seed)
        seed_elapsed = time.perf_counter() - seed_t0
        seed_mem_mb = memory_usage_bytes(seeds) / 1e6
        print(f"Generated {len(seeds)} seeds in {seed_elapsed:.2f} s (~{seed_mem_mb:.2f} MB array)")
        seeds_filename = f"seeds_{n_decks}.npy"
        save_seeds(seeds, seeds_filename)

        if RUN_SCORING:
            print("\n[Seed pipeline] computing scores (this may take a while)...")

            def example_score_fn(deck):
                return np.sum(deck)

            score_t0 = time.perf_counter()
            scores = compute_scores_from_seeds(seeds, example_score_fn)
            score_elapsed = time.perf_counter() - score_t0
            score_mem_mb = memory_usage_bytes(scores) / 1e6
            print(f"Computed scores for {scores.shape[0]} decks in {score_elapsed:.2f} s (~{score_mem_mb:.2f} MB array)")
            scores_filename = f"scores_{n_decks}.npy"
            save_scores(scores, scores_filename)
            del scores

        del seeds


if __name__ == "__main__":
    run_tests()



#Test Results 1 ( n_decks = 2_000_000)


# (base) matthewplambeck@Matthews-MacBook-Pro-2 src_EP % cd /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows
# uv run python Card_Game/src_EP/run_tests.py


# === Direct 2M deck test ===

# [Deck pipeline]
# get_decks was called with:
# Positional arguments:
#  (2000000, 12345)
# Keyword arguments:
#  {}
# get_decks ran for 0:00:01.940575
# Generated 2000000 decks in 1.94 s (~832.00 MB array)
# Saved chunk 1/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_0.npy
# Saved chunk 2/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_1.npy
# Saved chunk 3/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_2.npy
# Saved chunk 4/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_3.npy
# Saved chunk 5/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_4.npy
# Saved chunk 6/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_5.npy
# Saved chunk 7/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_6.npy
# Saved chunk 8/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_7.npy
# Saved chunk 9/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_8.npy
# Saved chunk 10/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_9.npy
# Saved chunk 11/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_10.npy
# Saved chunk 12/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_11.npy
# Saved chunk 13/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_12.npy
# Saved chunk 14/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_13.npy
# Saved chunk 15/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_14.npy
# Saved chunk 16/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_15.npy
# Saved chunk 17/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_16.npy
# Saved chunk 18/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_17.npy
# Saved chunk 19/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_18.npy
# Saved chunk 20/20 to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_19.npy
# Saved seed to /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_seed.npy
# [time_and_size] save_decks elapsed: 246.13 ms
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_0.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_1.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_2.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_3.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_4.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_5.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_6.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_7.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_8.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_9.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_10.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_11.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_12.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_13.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_14.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_15.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_16.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_17.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_18.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_2000000_19.npy (41600128 bytes)
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/decks_seed.npy (136 bytes)

# [Seed pipeline]
# Generated 2000000 seeds in 0.02 s (~16.00 MB array)
# [time_and_size] save_seeds elapsed: 8.46 ms
# [time_and_size] saved: /Users/matthewplambeck/Desktop/DATA_440_Automation_And_Workflows/Card_Game/data/seeds_2000000.npy (16000128 bytes)




# 