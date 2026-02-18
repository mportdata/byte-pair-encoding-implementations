from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import logging
import time
import re
import tracemalloc

# logging.basicConfig(
#     level=logging.DEBUG,
#     force=True
# )

logger = logging.getLogger(__name__)

def load_text_data() -> str:
    text_path = Path(__file__).resolve().parent.parent / "text.txt"
    text_data = text_path.read_text(encoding="utf-8")
    logger.info("Loaded text data (%d characters)", len(text_data))
    logger.debug("First 100 characters: %s", text_data[:100])
    return text_data

def _encode_types_iter(s: str) -> Iterable[tuple[int, ...]]:
    first_word = True
    for m in re.compile(r"\S+").finditer(s):
        w = m.group(0)
        if first_word:
            yield tuple(w.encode("utf-8"))  # first word: no leading space
            first_word = False
        else:
            yield tuple(("Ġ" + w).encode("utf-8"))  # subsequent words: add leading space

def build_type_counter(s: str) -> Counter[tuple[int, ...]]:
    counter: Counter[tuple[int, ...]] = Counter(_encode_types_iter(s))
    logger.debug("Counted types, total unique types: %d", len(counter))
    logger.debug("Most common 10 types as byte sequences: %s", counter.most_common(10))
    return counter

def build_pair_counter(types_counter: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    pair_counter: Counter[tuple[int, int]] = Counter()
    for type_tuple, count in types_counter.items():
        if len(type_tuple) < 2:
            continue
        pairs = zip(type_tuple, type_tuple[1:])
        for pair in pairs:
            pair_counter[pair] += count
    logger.debug("Counter pairs, total unique pairs: %d", len(pair_counter))
    logger.debug("Most common 10 pairs: %s", pair_counter.most_common(10))
    return pair_counter

def get_most_common_pair(counter: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    if not counter:
        return None
    most_common_pair, _ = counter.most_common(1)[0]
    logger.debug("Most common byte pair to merge: %s", most_common_pair)
    return most_common_pair

def apply_pair_merge(type_counter: Counter[tuple[int, ...]], pair: tuple[int, int], token: int) -> Counter[tuple[int, ...]]:
    new_counter = Counter()
    for type_tuple, count in type_counter.items():
        new_tuple = []
        i = 0
        while i < len(type_tuple):
            if i < len(type_tuple) - 1 and (type_tuple[i], type_tuple[i + 1]) == pair:
                new_tuple.append(token)
                i += 2  # Skip the next byte since it's part of the pair
            else:
                new_tuple.append(type_tuple[i])
                i += 1
        new_counter[tuple(new_tuple)] += count
    logger.debug("Replaced pair %s with token %d in decimal bytes tuple counter, total unique tuples: %d", pair, token, len(new_counter))
    return new_counter

def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")
    merge_dict: dict[tuple[int, int], int] = {}
    type_counter: Counter[tuple[int, ...]] = build_type_counter(text)
    pair_counter: Counter[tuple[int, int]] = build_pair_counter(type_counter)
    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair = get_most_common_pair(pair_counter)
        if most_common_pair is None:
            break
        merge_dict[most_common_pair] = base_vocab_size + len(merge_dict)
        type_counter: Counter[tuple[int, ...]] = apply_pair_merge(type_counter, most_common_pair, merge_dict[most_common_pair])
        pair_counter: Counter[tuple[int, int]] = build_pair_counter(type_counter)
        current_vocab_size = base_vocab_size + len(merge_dict)
    logger.info("Final vocabulary size: %d", current_vocab_size)
    return merge_dict
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    text_data: str = load_text_data()
    vocab_size: int = 257
    tracemalloc.start()
    t_0 = time.perf_counter()
    train_bpe(text_data, vocab_size)
    t_1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_seconds: float = t_1 - t_0
    logger.info("Training BPE took %.2f seconds", elapsed_seconds)
    logger.info(
        "Current memory usage: %.2f MB, Peak memory usage: %.2f MB",
        current / (1024 * 1024),
        peak / (1024 * 1024),
    )
    
