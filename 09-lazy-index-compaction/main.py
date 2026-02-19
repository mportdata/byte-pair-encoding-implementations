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
PAIR_MASK = (1 << 32) - 1
# Compaction is intentionally conservative to avoid adding overhead
# on medium-size runs where stale-index cleanup may not amortize.
COMPACTION_SIZE_THRESHOLD = 2048
COMPACTION_MIN_TOUCHED_FOR_WASTE = 512
COMPACTION_LIVE_RATIO_THRESHOLD = 0.25


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
            pair_to_type_ids.setdefault(pair_key, []).append(type_id)
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


def _find_pair_positions_in_type(type_seq: list[int], pair_key: int) -> list[int]:
    if len(type_seq) < 2:
        return []
    a, b = unpack_pair(pair_key)
    positions: list[int] = []
    i = 0
    while i < len(type_seq) - 1:
        if type_seq[i] == a and type_seq[i + 1] == b:
            positions.append(i)
            i += 2  # Skip the next byte since it's part of the pair
        else:
            i += 1
    return positions

def _alter_specific_pair_counters(
    local_counter: Counter[int],
    global_counter: Counter[int],
    pair_key: int,
    type_frequency: int,
    delta: int,
) -> None:
    # local
    new_local = local_counter.get(pair_key, 0) + delta
    if new_local:
        local_counter[pair_key] = new_local
    else:
        local_counter.pop(pair_key, None)

    # global (weighted)
    new_global = global_counter.get(pair_key, 0) + delta * type_frequency
    if new_global:
        global_counter[pair_key] = new_global
    else:
        global_counter.pop(pair_key, None)

def _build_windowed_delta_for_positions(
    type_seq: list[int],
    positions: list[int],
    pair_key: int,
    token: int,
) -> Counter[int]:
    a, b = unpack_pair(pair_key)
    deltas: Counter[int] = Counter()
    n = len(type_seq)

    for pos_idx, index in enumerate(positions):
        prev_adjacent = pos_idx > 0 and positions[pos_idx - 1] == index - 2
        next_adjacent = pos_idx + 1 < len(positions) and positions[pos_idx + 1] == index + 2

        left_exists = index - 1 >= 0
        right_exists = index + 2 < n

        if left_exists and not prev_adjacent:
            left = type_seq[index - 1]
            deltas[pack_pair(left, a)] -= 1
            deltas[pack_pair(left, token)] += 1

        deltas[pair_key] -= 1

        if right_exists and not next_adjacent:
            right = type_seq[index + 2]
            deltas[pack_pair(b, right)] -= 1
            deltas[pack_pair(token, right)] += 1

        if next_adjacent:
            deltas[pack_pair(token, token)] += 1

    return deltas

def compact_pair_index_for_key(
    pair_to_type_ids: dict[int, list[int]],
    pairs_by_type_id: list[Counter[int]],
    pair_key: int,
) -> tuple[int, int]:
    index_list = pair_to_type_ids.get(pair_key, [])
    if not index_list:
        return 0, 0

    original_size = len(index_list)
    seen: set[int] = set()
    compacted: list[int] = []

    for type_id in index_list:
        if type_id in seen:
            continue
        seen.add(type_id)
        if pairs_by_type_id[type_id].get(pair_key, 0) > 0:
            compacted.append(type_id)

    if compacted:
        pair_to_type_ids[pair_key] = compacted
    else:
        del pair_to_type_ids[pair_key]

    return original_size, len(compacted)

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
    if not affected_type_ids:
        return type_seqs, pairs_by_type_id, pair_to_type_ids, pair_counter
    touched = len(affected_type_ids)
    live = 0

    for type_id in affected_type_ids:
        if pairs_by_type_id[type_id].get(pair_key, 0) == 0:
            continue
        live += 1

        positions = _find_pair_positions_in_type(type_seqs[type_id], pair_key)
        if not positions:
            continue
        type_seq = type_seqs[type_id]
        local_counter = pairs_by_type_id[type_id]
        deltas = _build_windowed_delta_for_positions(type_seq, positions, pair_key, token)

        for changed_pair_key, delta in deltas.items():
            if delta == 0:
                continue
            _alter_specific_pair_counters(
                local_counter=local_counter,
                global_counter=pair_counter,
                pair_key=changed_pair_key,
                type_frequency=type_freqs[type_id],
                delta=delta,
            )
            if delta > 0:
                pair_to_type_ids.setdefault(changed_pair_key, []).append(type_id)

        new_type_seq: list[int] = []
        prev = 0
        for index in positions:
            new_type_seq.extend(type_seq[prev:index])
            new_type_seq.append(token)
            prev = index + 2
        new_type_seq.extend(type_seq[prev:])
        type_seqs[type_id] = new_type_seq

    should_compact = (
        touched >= COMPACTION_SIZE_THRESHOLD
        or (
            touched >= COMPACTION_MIN_TOUCHED_FOR_WASTE
            and (live / touched) <= COMPACTION_LIVE_RATIO_THRESHOLD
        )
    )
    if should_compact:
        before, after = compact_pair_index_for_key(pair_to_type_ids, pairs_by_type_id, pair_key)
        logger.debug(
            "Compacted pair index for key %s: %d -> %d entries (live/touched=%d/%d)",
            pair_key,
            before,
            after,
            live,
            touched,
        )

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
