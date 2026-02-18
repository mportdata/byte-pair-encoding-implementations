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
TOKEN_RE = re.compile(r"\S+")
PAIR_MASK = (1 << 32) - 1


def pack_pair(a: int, b: int) -> int:
    return (a << 32) | b


def unpack_pair(pair_key: int) -> tuple[int, int]:
    return pair_key >> 32, pair_key & PAIR_MASK


def load_text_data() -> str:
    text_path = Path(__file__).resolve().parent.parent / "text.txt"
    text_data = text_path.read_text(encoding="utf-8")
    logger.info("Loaded text data (%d characters)", len(text_data))
    logger.debug("First 100 characters: %s", text_data[:100])
    return text_data


def _encode_types_iter(s: str) -> Iterable[bytes]:
    first_word = True
    for m in TOKEN_RE.finditer(s):
        w = m.group(0)
        if first_word:
            yield w.encode("utf-8")  # first word: no leading space
            first_word = False
        else:
            yield ("Ġ" + w).encode("utf-8")  # subsequent words: add leading space


def _build_pairs_by_type_id(type_seqs: list[list[int]]) -> list[Counter[int]]:
    pairs_by_type_id: list[Counter[int]] = []

    for seq in type_seqs:
        if len(seq) < 2:
            pairs_by_type_id.append(Counter())
        else:
            pairs_by_type_id.append(Counter(pack_pair(a, b) for a, b in zip(seq, seq[1:])))

    return pairs_by_type_id


def build_type_state(s: str) -> tuple[list[list[int]], list[int], list[Counter[int]]]:
    counter: Counter[bytes] = Counter(_encode_types_iter(s))
    logger.debug("Counted types, total unique types: %d", len(counter))
    logger.debug("Most common 10 types as byte sequences: %s", counter.most_common(10))

    # Convert to mutable sequences (lists) because we will merge in-place later
    type_seqs: list[list[int]] = [list(b) for b in counter.keys()]

    # Keep frequencies aligned with type_seqs by iterating values in the same order as keys()
    type_freqs: list[int] = list(counter.values())

    pairs_by_type_id: list[Counter[int]] = _build_pairs_by_type_id(type_seqs)
    return type_seqs, type_freqs, pairs_by_type_id

def build_pair_to_type_ids(pairs_by_type_id: list[Counter[int]]) -> dict[int, list[int]]:
    pair_to_type_ids: dict[int, list[int]] = {}
    for type_id, counter in enumerate(pairs_by_type_id):
        for pair_key in counter:
            if pair_key not in pair_to_type_ids:
                pair_to_type_ids[pair_key] = []
            pair_to_type_ids[pair_key].append(type_id)
    return pair_to_type_ids

def _count_pairs_in_type(seq: list[int]) -> Counter[int]:
    return Counter(pack_pair(a, b) for a, b in zip(seq, seq[1:])) if len(seq) >= 2 else Counter()


def build_pair_counter(type_seqs: list[list[int]], type_freqs: list[int]) -> Counter[int]:
    pair_counter: Counter[int] = Counter()

    for seq, freq in zip(type_seqs, type_freqs):
        local = _count_pairs_in_type(seq)
        if not local:
            continue

        for pair_key, count in local.items():
            pair_counter[pair_key] += count * freq

    logger.debug("Counted pairs, total unique pairs: %d", len(pair_counter))
    logger.debug("Most common 10 pairs: %s", pair_counter.most_common(10))
    return pair_counter


def get_most_common_pair(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    most_common_pair_key, _ = counter.most_common(1)[0]
    logger.debug("Most common byte pair key to merge: %s", most_common_pair_key)
    return most_common_pair_key


def _update_pair_in_type_seq(type_seq: list[int], pair_key: int, token: int) -> list[int]:
    a, b = unpack_pair(pair_key)
    new_seq: list[int] = []
    i = 0
    while i < len(type_seq):
        if i < len(type_seq) - 1 and type_seq[i] == a and type_seq[i + 1] == b:
            new_seq.append(token)
            i += 2  # Skip the next byte since it's part of the pair
        else:
            new_seq.append(type_seq[i])
            i += 1
    return new_seq

def _apply_weighted_local_to_global(
    global_pairs: Counter[int],
    local_pairs: Counter[int],
    type_frequency: int,
    sign: int,  # -1 for subtract, +1 for add
) -> None:
    for pair_key, count in local_pairs.items():
        global_pairs[pair_key] += sign * count * type_frequency
        if global_pairs[pair_key] == 0:
            del global_pairs[pair_key]

def apply_pair_merge(
    type_seqs: list[list[int]],
    type_freqs: list[int],
    pairs_by_type_id: list[Counter[int]],
    pair_to_type_ids: dict[int, list[int]],
    pair_counter: Counter[int],
    pair_key: int,
    token: int,
) -> tuple[list[list[int]], list[Counter[int]], dict[int, list[int]], Counter[int]]:
    affected_type_ids = pair_to_type_ids.get(pair_key, [])
    for type_id in affected_type_ids:
        old_local_counter = pairs_by_type_id[type_id]
        if old_local_counter.get(pair_key, 0) == 0:
            continue

        _apply_weighted_local_to_global(pair_counter, old_local_counter, type_freqs[type_id], -1)

        new_seq = _update_pair_in_type_seq(type_seqs[type_id], pair_key, token)
        type_seqs[type_id] = new_seq

        new_local_counter = _count_pairs_in_type(new_seq)
        pairs_by_type_id[type_id] = new_local_counter
        
        _apply_weighted_local_to_global(pair_counter, new_local_counter, type_freqs[type_id], +1)

        for new_pair_key in new_local_counter:
            if new_pair_key not in old_local_counter:
                pair_to_type_ids.setdefault(new_pair_key, []).append(type_id)

    return type_seqs, pairs_by_type_id, pair_to_type_ids, pair_counter


def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")

    merge_dict: dict[tuple[int, int], int] = {}

    type_seqs, type_freqs, pairs_by_type_id = build_type_state(text)
    pair_to_type_ids = build_pair_to_type_ids(pairs_by_type_id)

    pair_counter: Counter[int] = build_pair_counter(type_seqs, type_freqs)

    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair_key = get_most_common_pair(pair_counter)
        if most_common_pair_key is None:
            break

        token = base_vocab_size + len(merge_dict)
        pair = unpack_pair(most_common_pair_key)
        merge_dict[pair] = token

        type_seqs, pairs_by_type_id, pair_to_type_ids, pair_counter = apply_pair_merge(
            type_seqs=type_seqs,
            type_freqs=type_freqs,
            pairs_by_type_id=pairs_by_type_id,
            pair_to_type_ids=pair_to_type_ids,
            pair_counter=pair_counter,
            pair_key=most_common_pair_key,
            token=token,
        )

        current_vocab_size = base_vocab_size + len(merge_dict)

    logger.info("Final vocabulary size: %d", current_vocab_size)
    return merge_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    text_data: str = load_text_data()
    vocab_size: int = 270
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
