# Byte Pair Encoding Implementations

A repo of byte pair encoding implementations, in order of increasing efficiency

## 1. Naive Implementation

A basic implementation of the BPE algorithm in Python.

## 2. Type Frequency

In this implementation we break the corpus into types (here this is words with punctuation, special characters and any leading white-space represented as Ġ).

## 3. Stream Type Counting

In this implementation we don't materialise the large words list and the large types list, instead we yield types one by one directly into the counter.