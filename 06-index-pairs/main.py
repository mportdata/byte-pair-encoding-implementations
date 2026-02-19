from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import logging
import time
import re
import tracemalloc
import yaml

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\S+")


def load_text_data() -> str:
    text_path = Path(__file__).resolve().parent.parent / "text.txt"
    text_data = text_path.read_text(encoding="utf-8")
    logger.info("Loaded text data (%d characters)", len(text_data))
    logger.debug("First 100 characters: %s", text_data[:100])
    return text_data

def load_vocab_size() -> int:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    vocab_size = config_data.get("vocab_size")
    if not isinstance(vocab_size, int):
        raise ValueError("config.yaml must define integer vocab_size")
    return vocab_size

def configure_logging() -> None:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    debug_logging = config_data.get("debug_logging", False)
    if not isinstance(debug_logging, bool):
        raise ValueError("config.yaml debug_logging must be a boolean")
    level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def _encode_types_iter(s: str) -> Iterable[tuple[int, ...]]:
    first_word = True
    for m in TOKEN_RE.finditer(s):
        w = m.group(0)
        if first_word:
            yield tuple(w.encode("utf-8"))  # first word: no leading space
            first_word = False
        else:
            yield tuple(("Ġ" + w).encode("utf-8"))  # subsequent words: add leading space


def _build_pairs_by_type_id(type_seqs: list[list[int]]) -> list[Counter[tuple[int, int]]]:
    pairs_by_type_id: list[Counter[tuple[int, int]]] = []

    for seq in type_seqs:
        if len(seq) < 2:
            pairs_by_type_id.append(Counter())
        else:
            pairs_by_type_id.append(Counter(zip(seq, seq[1:])))

    return pairs_by_type_id


def build_type_state(s: str) -> tuple[list[list[int]], list[int], list[Counter[tuple[int, int]]]]:
    counter: Counter[tuple[int, ...]] = Counter(_encode_types_iter(s))
    logger.debug("Counted types, total unique types: %d", len(counter))
    logger.debug("Most common 10 types as byte sequences: %s", counter.most_common(10))

    # Convert to mutable sequences (lists) because we will merge in-place later
    type_seqs: list[list[int]] = [list(t) for t in counter.keys()]

    # Keep frequencies aligned with type_seqs by iterating values in the same order as keys()
    type_freqs: list[int] = list(counter.values())

    pairs_by_type_id: list[Counter[tuple[int, int]]] = _build_pairs_by_type_id(type_seqs)
    return type_seqs, type_freqs, pairs_by_type_id

def build_pair_to_type_ids(pairs_by_type_id: list[Counter[tuple[int, int]]]) -> dict[tuple[int, int], set[int]]:
    pair_to_type_ids: dict[tuple[int, int], set[int]] = {}
    for type_id, counter in enumerate(pairs_by_type_id):
        for pair in counter:
            if pair not in pair_to_type_ids:
                pair_to_type_ids[pair] = set()
            pair_to_type_ids[pair].add(type_id)
    return pair_to_type_ids

def _count_pairs_in_type(seq: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(seq, seq[1:])) if len(seq) >= 2 else Counter()


def build_pair_counter(type_seqs: list[list[int]], type_freqs: list[int]) -> Counter[tuple[int, int]]:
    pair_counter: Counter[tuple[int, int]] = Counter()

    for seq, freq in zip(type_seqs, type_freqs):
        local = _count_pairs_in_type(seq)
        if not local:
            continue

        for pair, c in local.items():
            pair_counter[pair] += c * freq

    logger.debug("Counted pairs, total unique pairs: %d", len(pair_counter))
    logger.debug("Most common 10 pairs: %s", pair_counter.most_common(10))
    return pair_counter


def get_most_common_pair(counter: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    if not counter:
        return None
    most_common_pair, _ = counter.most_common(1)[0]
    logger.debug("Most common byte pair to merge: %s", most_common_pair)
    return most_common_pair


def _update_pair_in_type_seq(type_seq: list[int], pair: tuple[int, int], token: int) -> list[int]:
    new_seq: list[int] = []
    i = 0
    while i < len(type_seq):
        if i < len(type_seq) - 1 and (type_seq[i], type_seq[i + 1]) == pair:
            new_seq.append(token)
            i += 2  # Skip the next byte since it's part of the pair
        else:
            new_seq.append(type_seq[i])
            i += 1
    return new_seq

def _apply_weighted_local_to_global(
    global_pairs: Counter[tuple[int, int]],
    local_pairs: Counter[tuple[int, int]],
    type_frequency: int,
    sign: int,  # -1 for subtract, +1 for add
) -> None:
    for pair, count in local_pairs.items():
        global_pairs[pair] += sign * count * type_frequency
        if global_pairs[pair] == 0:
            del global_pairs[pair]

def apply_pair_merge(
    type_seqs: list[list[int]],
    type_freqs: list[int],
    pairs_by_type_id: list[Counter[tuple[int, int]]],
    pair_to_type_ids: dict[tuple[int, int], set[int]],
    pair_counter: Counter[tuple[int, int]],
    pair: tuple[int, int],
    token: int,
) -> tuple[list[list[int]], list[Counter[tuple[int, int]]], dict[tuple[int, int], set[int]], Counter[tuple[int, int]]]:
    affected_type_ids = list(pair_to_type_ids.get(pair, set()))
    for type_id in affected_type_ids:
        old_local_counter = pairs_by_type_id[type_id]
        if old_local_counter.get(pair, 0) == 0:
            continue

        _apply_weighted_local_to_global(pair_counter, old_local_counter, type_freqs[type_id], -1)

        new_seq = _update_pair_in_type_seq(type_seqs[type_id], pair, token)
        type_seqs[type_id] = new_seq

        new_local_counter = _count_pairs_in_type(new_seq)
        pairs_by_type_id[type_id] = new_local_counter
        
        _apply_weighted_local_to_global(pair_counter, new_local_counter, type_freqs[type_id], +1)

        for old_pair in old_local_counter:
            if old_pair not in new_local_counter:
                type_ids = pair_to_type_ids.get(old_pair)
                if type_ids is not None:
                    type_ids.discard(type_id)
                    if not type_ids:
                        del pair_to_type_ids[old_pair]

        for new_pair in new_local_counter:
            if new_pair not in old_local_counter:
                pair_to_type_ids.setdefault(new_pair, set()).add(type_id)

    return type_seqs, pairs_by_type_id, pair_to_type_ids, pair_counter


def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")

    merge_dict: dict[tuple[int, int], int] = {}

    type_seqs, type_freqs, pairs_by_type_id = build_type_state(text)
    pair_to_type_ids = build_pair_to_type_ids(pairs_by_type_id)

    pair_counter: Counter[tuple[int, int]] = build_pair_counter(type_seqs, type_freqs)

    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair = get_most_common_pair(pair_counter)
        if most_common_pair is None:
            break

        token = base_vocab_size + len(merge_dict)
        merge_dict[most_common_pair] = token

        type_seqs, pairs_by_type_id, pair_to_type_ids, pair_counter = apply_pair_merge(
            type_seqs=type_seqs,
            type_freqs=type_freqs,
            pairs_by_type_id=pairs_by_type_id,
            pair_to_type_ids=pair_to_type_ids,
            pair_counter=pair_counter,
            pair=most_common_pair,
            token=token,
        )

        current_vocab_size = base_vocab_size + len(merge_dict)

    logger.info("Final vocabulary size: %d", current_vocab_size)
    return merge_dict


if __name__ == "__main__":
    configure_logging()
    text_data: str = load_text_data()
    vocab_size: int = load_vocab_size()
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
