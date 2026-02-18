from collections import Counter
from pathlib import Path

import logging
import time
import tracemalloc

logger = logging.getLogger(__name__)

# logging.basicConfig(
#     level=logging.DEBUG,
#     force=True
# )

def load_text_data() -> str:
    text_path = Path(__file__).resolve().parent.parent / "text.txt"
    text_data = text_path.read_text(encoding="utf-8")
    logger.info("Loaded text data (%d characters)", len(text_data))
    logger.debug("First 100 characters: %s", text_data[:100])
    return text_data

def bytes_list(s: str) -> list[int]:
    text_bytes = s.encode("utf-8")
    text_bytes_list = list(text_bytes)
    logger.info("Converted text to bytes list (%d bytes)", len(text_bytes_list))
    logger.debug("First 10 bytes: %s", text_bytes_list[:10])
    return text_bytes_list

def build_pair_counter(bytes_list: list[int]) -> Counter[tuple[int, int]]:
    pairs = zip(bytes_list, bytes_list[1:])
    counter = Counter(pairs)
    logger.debug("Counted byte pairs, total unique pairs: %d", len(counter))
    logger.debug("Most common 10 byte pairs: %s", counter.most_common(10))
    return counter

def get_most_common(counter: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    if not counter:
        return None
    most_common_pair, _ = counter.most_common(1)[0]
    logger.debug("Most common byte pair to merge: %s", most_common_pair)
    return most_common_pair

def update_pair_in_bytes_list(bytes_list: list[int], pair: tuple[int, int], token: int) -> list[int]:
    new_bytes_list = []
    i = 0
    while i < len(bytes_list):
        if i < len(bytes_list) - 1 and (bytes_list[i], bytes_list[i + 1]) == pair:
            new_bytes_list.append(token)
            i += 2  # Skip the next byte since it's part of the pair
        else:
            new_bytes_list.append(bytes_list[i])
            i += 1
    logger.debug("Replaced pair %s with token %d, new length: %d", pair, token, len(new_bytes_list))
    return new_bytes_list

def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")
    merge_dict: dict[tuple[int, int], int] = {}
    text_bytes: list[int] = bytes_list(text)
    byte_pairs_counter: Counter[tuple[int, int]] = build_pair_counter(text_bytes)
    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair: tuple[int, int] | None = get_most_common(byte_pairs_counter)
        if most_common_pair is None:
            break
        merge_dict[most_common_pair] = base_vocab_size + len(merge_dict)
        text_bytes: list[int] = update_pair_in_bytes_list(text_bytes, most_common_pair, merge_dict[most_common_pair])
        byte_pairs_counter: Counter[tuple[int, int]] = build_pair_counter(text_bytes)
        current_vocab_size: int = base_vocab_size + len(merge_dict)
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

    elapsed_seconds = t_1 - t_0
    logger.info("Training BPE took %.2f seconds", elapsed_seconds)
    logger.info(
        "Current memory usage: %.2f MB, Peak memory usage: %.2f MB",
        current / (1024 * 1024),
        peak / (1024 * 1024),
    )
    
