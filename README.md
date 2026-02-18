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
