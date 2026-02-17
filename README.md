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
