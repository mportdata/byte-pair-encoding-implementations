# Byte Pair Encoding Implementations

A collection of byte pair encoding implementations, presented in order
of increasing efficiency.

## 1. Naive Implementation

A direct, unoptimized implementation of the BPE algorithm in Python.

## 2. Type Frequency

In this implementation, we break the corpus into types (here: words with
punctuation and special characters, with leading whitespace represented
as `Ġ`) and count their frequencies before applying BPE.

This avoids repeating work on identical tokens.

## 3. Stream Type Counting

In this implementation, we avoid materialising large intermediate lists
of words and types. Instead, we scan the corpus and yield types one by one
directly into a Counter.

This significantly reduces peak memory usage.

## 4. Stream Type Counting (Regex-Based)

In this implementation, we keep the streaming approach from Section 3
but move the text scanning into Python’s C-based regular expression
engine using `re.compile` and `finditer`.

This preserves the memory savings of streaming while recovering much of
the performance lost to Python-level character iteration.

## 5. Incremental Pair Updates

In this implementation, we avoid rebuilding the full global pair counter
after every merge. Instead, we keep per-type pair counts and update the
global counts incrementally only for types affected by the selected merge.

This preserves correctness while substantially reducing repeated work
across merge iterations.

## 6. Pair-to-Type Index

In this implementation, we add an index from each pair to the set of type IDs
that currently contain it. During each merge step, we update only those indexed
types instead of scanning all types.

This keeps the incremental-update strategy from Section 5 while reducing
per-merge work further as the vocabulary grows.

## 7. Memory-Lean Representations

In this implementation, we replace higher-overhead Python objects with
memory-lean equivalents while preserving behavior:
- represent types as `bytes` values instead of tuples of ints
- represent pairs as packed integer keys (`pack_pair(a, b)`)
- use a lazy list-based pair-to-type index (`dict[pair_key, list[type_id]]`)

The list index is append-only and may contain stale or duplicate type IDs, but
correctness is preserved by checking pair presence in each type before applying
updates. This reduces memory usage while keeping incremental pair-count updates.

## 8. Windowed Pair Updates (Locality)

In this implementation, we avoid rebuilding full per-type pair counters after a
merge. Instead, for each affected type we update pair counts using local
windowed deltas around replacement positions and rebuild the merged sequence in
one pass.

This keeps correctness while reducing repeated per-type full scans and lowering
allocation churn during training.

## 9. Lazy Index Compaction

In this step, we add compaction/maintenance for lazy pair-to-type indices to
control stale or duplicate entries over time while preserving correctness.

Current compaction thresholds in `09-lazy-index-compaction/main.py`:
- `COMPACTION_SIZE_THRESHOLD = 2048`
- `COMPACTION_MIN_TOUCHED_FOR_WASTE = 512`
- `COMPACTION_LIVE_RATIO_THRESHOLD = 0.25`

Observed benchmark note:
- with `max_mb: 50` and `vocab_size: 500`, these settings showed a runtime
  improvement versus the non-compacting lazy-index variant.

## 10. Heap Pair Selection

In this implementation, we replace repeated full-map max selection with a
max-heap (implemented via Python's min-heap + negative counts) and lazy
invalidation of stale entries.

Changed pair keys are pushed back to the heap after each merge, and a stale-heap
rebuild guard keeps heap growth under control during long runs.

Observed benchmark note:
- with `max_mb: 100` and `vocab_size: 2000`:
  - Step 09: `123.14s` (peak memory `543.58 MB`)
  - Step 10: `117.88s` (peak memory `544.49 MB`)

## 11. Sequence Storage Optimization

In this implementation, we replace per-type Python `list[int]` token
sequences with `array('I')` (unsigned int arrays) to reduce object overhead
while preserving the same merge and counter-update behavior.

This keeps algorithmic behavior unchanged while making sequence storage more
compact in hot update paths.

Observed benchmark note:
- with `max_mb: 100` and `vocab_size: 2000`:
  - Step 10: `117.88s` (peak memory `544.49 MB`)
  - Step 11: `124.84s` (peak memory `535.95 MB`)

## 12. Batch Delta Application

In this implementation, we keep windowed merge locality from step 8 but apply
local and global counter changes in a single batched pass per affected type.
Instead of repeatedly mutating counters for each individual pair-touch, we
first aggregate net deltas and then apply them once.

This reduces Python-loop and dictionary-mutation overhead in the hot merge path
while preserving correctness.

Observed benchmark notes:
- with `max_mb: 100`, `vocab_size: 2000`, `debug_logging: true`:
  - Step 11: `124.84s` (peak memory `535.95 MB`)
  - Step 12: `135.02s` (peak memory `535.95 MB`)
- with `max_mb: 500`, `vocab_size: 10000`, `debug_logging: true`:
  - Step 11: `2915.65s` (peak memory `3176.38 MB`)
  - Step 12: `3105.91s` (peak memory `3176.38 MB`)

Conclusion:
- this batching tradeoff did not improve performance in tested configs and is
  kept as a documented experiment, but not used as the baseline for later steps.

## 13. Adaptive Compaction Policy

In this implementation, compaction decisions are made by an adaptive policy
instead of fixed thresholds. The policy uses:
- `touched` vs `live` ratio for staleness/waste
- per-key cooldown to avoid over-compacting hot keys
- rolling compaction payoff (`shrink_history`) to tune aggressiveness over time

This aims to reduce unnecessary compaction work while still cleaning stale
index entries when it is likely to pay off.

Observed benchmark note:
- with `max_mb: 100`, `vocab_size: 10000`, `debug_logging: true`:
  - Step 11: `325.76s` (peak memory `940.05 MB`)
  - Step 13: `279.17s` (peak memory `938.19 MB`)

Interpretation:
- in this high-vocab setting, adaptive compaction improved runtime by ~14.3%
  while keeping peak memory effectively flat.

## 14. Structure of Arrays `TODO`

Explore a structure-of-arrays representation for core training state to improve
cache behavior and reduce object overhead in large runs.
