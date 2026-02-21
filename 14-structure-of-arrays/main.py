from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from array import array

import logging
import time
import re
import tracemalloc
import yaml
import heapq

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\S+")
PAIR_MASK = (1 << 32) - 1

INITIAL_COMPACTION_MIN_SIZE = 2048
INITIAL_COMPACTION_LIVE_RATIO_THRESHOLD = 0.25
COMPACTION_MIN_SIZE_FLOOR = 256
COMPACTION_MIN_SIZE_CEIL = 32768
COMPACTION_LIVE_RATIO_FLOOR = 0.05
COMPACTION_LIVE_RATIO_CEIL = 0.60
COMPACTION_COOLDOWN_MERGES = 64
COMPACTION_HISTORY_WINDOW = 64
COMPACTION_MIN_HISTORY_FOR_ADAPT = 8
COMPACTION_STRONG_SHRINK = 0.45
COMPACTION_WEAK_SHRINK = 0.15


def pack_pair(a: int, b: int) -> int:
    return (a << 32) | b


def unpack_pair(pair_key: int) -> tuple[int, int]:
    return pair_key >> 32, pair_key & PAIR_MASK

@dataclass
class CompactionPolicy:
    min_size: int = INITIAL_COMPACTION_MIN_SIZE
    live_ratio_threshold: float = INITIAL_COMPACTION_LIVE_RATIO_THRESHOLD
    cooldown_merges: int = COMPACTION_COOLDOWN_MERGES
    last_compacted_merge_by_key: dict[int, int] = field(default_factory=dict)
    shrink_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=COMPACTION_HISTORY_WINDOW)
    )

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


def build_sequence_soa(type_bytes: Iterable[bytes]) -> tuple[array, array]:
    seq_data = array("I")
    seq_offsets = array("I", [0])
    for b in type_bytes:
        seq_data.extend(byte for byte in b)
        seq_offsets.append(len(seq_data))
    return seq_data, seq_offsets


def get_type_seq(seq_data: array, seq_offsets: array, type_id: int) -> array:
    start = seq_offsets[type_id]
    end = seq_offsets[type_id + 1]
    return seq_data[start:end]


def replace_type_seq(
    seq_data: array,
    seq_offsets: array,
    type_id: int,
    new_type_seq: array,
) -> tuple[array, array]:
    start = seq_offsets[type_id]
    end = seq_offsets[type_id + 1]
    old_len = end - start
    new_len = len(new_type_seq)
    delta = new_len - old_len

    seq_data[start:end] = new_type_seq
    if delta != 0:
        for offset_idx in range(type_id + 1, len(seq_offsets)):
            seq_offsets[offset_idx] += delta

    return seq_data, seq_offsets


def _count_pairs_in_bounds(seq_data: array, start: int, end: int) -> Counter[int]:
    if end - start < 2:
        return Counter()
    return Counter(pack_pair(seq_data[i], seq_data[i + 1]) for i in range(start, end - 1))


def _build_pairs_by_type_id(seq_data: array, seq_offsets: array) -> list[Counter[int]]:
    pairs_by_type_id: list[Counter[int]] = []
    num_types = len(seq_offsets) - 1
    for type_id in range(num_types):
        start = seq_offsets[type_id]
        end = seq_offsets[type_id + 1]
        pairs_by_type_id.append(_count_pairs_in_bounds(seq_data, start, end))

    return pairs_by_type_id


def build_type_state(s: str) -> tuple[array, array, list[int], list[Counter[int]]]:
    counter: Counter[bytes] = Counter(_encode_types_iter(s))
    logger.debug("Counted types, total unique types: %d", len(counter))
    logger.debug("Most common 10 types as byte sequences: %s", counter.most_common(10))

    seq_data, seq_offsets = build_sequence_soa(counter.keys())

    # Keep frequencies aligned with seq_offsets by iterating values in key order
    type_freqs: list[int] = list(counter.values())

    pairs_by_type_id: list[Counter[int]] = _build_pairs_by_type_id(seq_data, seq_offsets)
    return seq_data, seq_offsets, type_freqs, pairs_by_type_id

def build_pair_to_type_ids(pairs_by_type_id: list[Counter[int]]) -> dict[int, list[int]]:
    pair_to_type_ids: dict[int, list[int]] = {}
    for type_id, counter in enumerate(pairs_by_type_id):
        for pair_key in counter:
            pair_to_type_ids.setdefault(pair_key, []).append(type_id)
    return pair_to_type_ids

def build_pair_counter(seq_data: array, seq_offsets: array, type_freqs: list[int]) -> Counter[int]:
    pair_counter: Counter[int] = Counter()

    for type_id, freq in enumerate(type_freqs):
        start = seq_offsets[type_id]
        end = seq_offsets[type_id + 1]
        local = _count_pairs_in_bounds(seq_data, start, end)
        if not local:
            continue

        for pair_key, count in local.items():
            pair_counter[pair_key] += count * freq

    logger.debug("Counted pairs, total unique pairs: %d", len(pair_counter))
    logger.debug("Most common 10 pairs: %s", pair_counter.most_common(10))
    return pair_counter

def build_pair_heap(pair_counter: Counter[int]) -> list[tuple[int, int]]:
    pair_heap = [(-count, pair_key) for pair_key, count in pair_counter.items()]
    heapq.heapify(pair_heap)
    return pair_heap

def pop_most_common_pair(pair_heap: list[tuple[int, int]], pair_counter: Counter[int]) -> int | None:
    while pair_heap:
        negative_count, pair_key = heapq.heappop(pair_heap)
        current_count = pair_counter.get(pair_key, 0)
        if current_count == 0:
            continue
        if -negative_count != current_count:
            continue
        logger.debug("Most common byte pair key to merge: %s", pair_key)
        return pair_key
    return None


def _find_pair_positions_in_type(type_seq: array, pair_key: int) -> list[int]:
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
    type_seq: array,
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

def should_compact_key(
    policy: CompactionPolicy,
    pair_key: int,
    merge_step: int,
    touched: int,
    live: int,
) -> bool:
    if touched < policy.min_size:
        return False

    last = policy.last_compacted_merge_by_key.get(pair_key)
    if last is not None and merge_step - last < policy.cooldown_merges:
        return False

    live_ratio = (live / touched) if touched else 1.0
    return live_ratio <= policy.live_ratio_threshold

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

def update_compaction_policy(
    policy: CompactionPolicy,
    pair_key: int,
    merge_step: int,
    before: int,
    after: int,
) -> None:
    policy.last_compacted_merge_by_key[pair_key] = merge_step
    if before <= 0:
        return

    shrink = (before - after) / before
    policy.shrink_history.append(shrink)

    if len(policy.shrink_history) < COMPACTION_MIN_HISTORY_FOR_ADAPT:
        return

    avg_shrink = sum(policy.shrink_history) / len(policy.shrink_history)

    if avg_shrink >= COMPACTION_STRONG_SHRINK:
        policy.min_size = max(
            COMPACTION_MIN_SIZE_FLOOR,
            int(policy.min_size * 0.9)
        )
        policy.live_ratio_threshold = min(
            COMPACTION_LIVE_RATIO_CEIL, 
            policy.live_ratio_threshold + 0.02
        )
    elif avg_shrink <= COMPACTION_WEAK_SHRINK:
        policy.min_size = min(
            COMPACTION_MIN_SIZE_CEIL, 
            int(policy.min_size * 1.1)
        )
        policy.live_ratio_threshold = max(
            COMPACTION_LIVE_RATIO_FLOOR, 
            policy.live_ratio_threshold - 0.02
        )

def apply_pair_merge(
    seq_data: array,
    seq_offsets: array,
    type_freqs: list[int],
    pairs_by_type_id: list[Counter[int]],
    pair_to_type_ids: dict[int, list[int]],
    pair_counter: Counter[int],
    pair_key: int,
    token: int,
    merge_step: int,
    compaction_policy: CompactionPolicy,
) -> tuple[array, array, list[Counter[int]], dict[int, list[int]], Counter[int], set[int]]:
    affected_type_ids = pair_to_type_ids.get(pair_key, [])
    if not affected_type_ids:
        return seq_data, seq_offsets, pairs_by_type_id, pair_to_type_ids, pair_counter, set()
    touched = len(affected_type_ids)
    live = 0
    changed_pair_keys: set[int] = set()

    for type_id in affected_type_ids:
        if pairs_by_type_id[type_id].get(pair_key, 0) == 0:
            continue
        live += 1

        type_seq = get_type_seq(seq_data, seq_offsets, type_id)
        positions = _find_pair_positions_in_type(type_seq, pair_key)
        if not positions:
            continue
        local_counter = pairs_by_type_id[type_id]
        deltas = _build_windowed_delta_for_positions(type_seq, positions, pair_key, token)

        for changed_pair_key, delta in deltas.items():
            if delta == 0:
                continue
            changed_pair_keys.add(changed_pair_key)
            _alter_specific_pair_counters(
                local_counter=local_counter,
                global_counter=pair_counter,
                pair_key=changed_pair_key,
                type_frequency=type_freqs[type_id],
                delta=delta,
            )
            if delta > 0:
                pair_to_type_ids.setdefault(changed_pair_key, []).append(type_id)

        new_type_seq = array("I")
        prev = 0
        for index in positions:
            new_type_seq.extend(type_seq[prev:index])
            new_type_seq.append(token)
            prev = index + 2
        new_type_seq.extend(type_seq[prev:])
        seq_data, seq_offsets = replace_type_seq(seq_data, seq_offsets, type_id, new_type_seq)

    should_compact = should_compact_key(
      policy=compaction_policy,
      pair_key=pair_key,
      merge_step=merge_step,
      touched=touched,
      live=live,
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
        update_compaction_policy(
            policy=compaction_policy,
            pair_key=pair_key,
            merge_step=merge_step,
            before=before,
            after=after,
        )

    return seq_data, seq_offsets, pairs_by_type_id, pair_to_type_ids, pair_counter, changed_pair_keys


def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    base_vocab_size = 256
    if vocab_size < base_vocab_size:
        raise ValueError(f"Vocabulary size must be at least {base_vocab_size}")

    merge_dict: dict[tuple[int, int], int] = {}

    seq_data, seq_offsets, type_freqs, pairs_by_type_id = build_type_state(text)
    pair_to_type_ids = build_pair_to_type_ids(pairs_by_type_id)

    pair_counter: Counter[int] = build_pair_counter(seq_data, seq_offsets, type_freqs)
    pair_heap = build_pair_heap(pair_counter)

    compaction_policy = CompactionPolicy()
    merge_step = 0

    current_vocab_size: int = base_vocab_size + len(merge_dict)
    while current_vocab_size < vocab_size:
        most_common_pair_key = pop_most_common_pair(pair_heap, pair_counter)
        if most_common_pair_key is None:
            break

        token = base_vocab_size + len(merge_dict)
        pair = unpack_pair(most_common_pair_key)
        if pair in merge_dict:
            pair_counter.pop(most_common_pair_key, None)
            continue
        merge_dict[pair] = token

        seq_data, seq_offsets, pairs_by_type_id, pair_to_type_ids, pair_counter, changed_pair_keys = apply_pair_merge(
            seq_data=seq_data,
            seq_offsets=seq_offsets,
            type_freqs=type_freqs,
            pairs_by_type_id=pairs_by_type_id,
            pair_to_type_ids=pair_to_type_ids,
            pair_counter=pair_counter,
            pair_key=most_common_pair_key,
            token=token,
            merge_step=merge_step,
            compaction_policy=compaction_policy,
        )
        for changed_pair_key in changed_pair_keys:
            new_count = pair_counter.get(changed_pair_key, 0)
            if new_count > 0:
                heapq.heappush(pair_heap, (-new_count, changed_pair_key))
        if len(pair_heap) > 4 * max(1, len(pair_counter)):
            pair_heap = build_pair_heap(pair_counter)

        merge_step += 1
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
