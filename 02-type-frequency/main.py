from collections import Counter
from pathlib import Path

import logging
import time
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

def _encode_types(s: str) -> list[list[bytes]]:
    words = s.split()   # split on any whitespace
    if not words:
        return []
    units = [tuple(words[0].encode("utf-8"))]     # first word: no leading space
    for w in words[1:]:
        units.append(tuple(("Ġ" + w).encode("utf-8")))  # subsequent words: add leading space
    logger.info("Converted text to types list (%d types)", len(units))
    logger.debug("First 10 byte encoded types: %s", units[:10])
    return units

def build_types_counter(s:str) -> Counter[tuple[int, ...]]:
    types_list: list[bytes] = _encode_types(s)
    counter: Counter[tuple[int, ...]] = Counter(types_list)
    logger.debug("Counted types, total unique types: %d", len(counter))
    logger.debug("Most common 10 types as byte sequences: %s", counter.most_common(10))
    return counter

def build_pair_counter(types_counter: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    decimal_bytes_pair_counter = Counter()
    for decimal_bytes_tuple, count in types_counter.items():
        if len(decimal_bytes_tuple) < 2:
            continue
        pairs = zip(decimal_bytes_tuple, decimal_bytes_tuple[1:])
        for pair in pairs:
            decimal_bytes_pair_counter[pair] += count
    logger.debug("Counted pairs, total unique pairs: %d", len(decimal_bytes_pair_counter))
    logger.debug("Most common 10 pairs: %s", decimal_bytes_pair_counter.most_common(10))
    return decimal_bytes_pair_counter

def get_most_common(counter: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    if not counter:
        return None
    most_common_pair, _ = counter.most_common(1)[0]
    logger.debug("Most common byte pair to merge: %s", most_common_pair)
    return most_common_pair

def update_pair_in_type_counter(type_counter: Counter[tuple[int, ...]], pair: tuple[int, int], token: int) -> Counter[tuple[int, ...]]:
    new_counter = Counter()
    for decimal_bytes_tuple, count in type_counter.items():
        new_tuple = []
        i = 0
        while i < len(decimal_bytes_tuple):
            if i < len(decimal_bytes_tuple) - 1 and (decimal_bytes_tuple[i], decimal_bytes_tuple[i + 1]) == pair:
                new_tuple.append(token)
                i += 2  # Skip the next byte since it's part of the pair
            else:
                new_tuple.append(decimal_bytes_tuple[i])
                i += 1
        new_counter[tuple(new_tuple)] += count
    logger.debug("Replaced pair %s with token %d in decimal bytes tuple counter, total unique tuples: %d", pair, token, len(new_counter))
    return new_counter

def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")
    merge_dict: dict[tuple[int, int], int] = {}
    type_counter: Counter[tuple[int, ...]] = build_types_counter(text)
    pair_counter: Counter[tuple[int, int]] = build_pair_counter(type_counter)
    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair = get_most_common(pair_counter)
        if most_common_pair is None:
            break
        merge_dict[most_common_pair] = base_vocab_size + len(merge_dict)
        type_counter: Counter[tuple[int, ...]] = update_pair_in_type_counter(type_counter, most_common_pair, merge_dict[most_common_pair])
        pair_counter: Counter[tuple[int, int]] = build_pair_counter(type_counter)
        current_vocab_size = base_vocab_size + len(merge_dict)
    logger.info("Final vocabulary size: %d", current_vocab_size)
    return merge_dict
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    text_data = load_text_data()
    vocab_size = 257
    tracemalloc.start()
    t_0 = time.perf_counter()
    train_bpe(text_data, vocab_size)
    t_1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_seconds = t_1 - t_0
    logger.info("Training BPE took %.2f seconds", elapsed_seconds)
    logger.info(
        "Current memory usage: %.2f MB, Peak memory usage: %.2f MB",
        current / (1024 * 1024),
        peak / (1024 * 1024),
    )
    
