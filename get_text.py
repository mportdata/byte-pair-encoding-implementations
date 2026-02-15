from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download, list_repo_files


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_text(repo_id: str, config_name: str, split: str, text_field: str):
    prefix = f"{config_name}/{split}-"
    files = [
        name
        for name in list_repo_files(repo_id=repo_id, repo_type="dataset")
        if name.startswith(prefix) and name.endswith(".parquet")
    ]

    if not files:
        raise RuntimeError(
            f"No parquet files found for prefix '{prefix}' in dataset '{repo_id}'."
        )

    for filename in sorted(files):
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(columns=[text_field], batch_size=1024):
            for value in batch.column(0).to_pylist():
                text = str(value).strip() if value is not None else ""
                if text:
                    yield text


def main() -> None:
    cfg = load_config()

    dataset_name = cfg["dataset_name"]
    dataset_config = cfg["dataset_config"]
    split = cfg.get("split", "train")
    text_field = cfg.get("text_field", "text")
    max_mb = float(cfg["max_mb"])
    max_bytes = int(max_mb * 1024 * 1024)
    output_path = Path(cfg.get("output_path", "text.txt"))

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for text in iter_text(dataset_name, dataset_config, split, text_field):
            chunk = text
            chunk_bytes = len(chunk.encode("utf-8"))

            if written + chunk_bytes > max_bytes:
                remaining = max_bytes - written
                if remaining > 0:
                    trimmed_bytes = chunk.encode("utf-8")[:remaining]
                    trimmed = trimmed_bytes.decode("utf-8", errors="ignore")
                    f.write(trimmed)
                    written += len(trimmed.encode("utf-8"))
                break

            f.write(chunk)
            written += chunk_bytes

    print(f"Wrote {written} bytes to {output_path}")


if __name__ == "__main__":
    main()
