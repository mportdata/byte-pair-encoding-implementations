import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

def download_text(repo_id: str, config_name: str, split: str, max_mb: int):
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

    output_path = Path("data/text.txt")
    max_bytes = int(max_mb * 1024 * 1024)
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for filename in sorted(files):
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
            )
            parquet = pq.ParquetFile(local_path)
            for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
                for value in batch.column(0).to_pylist():
                    text = str(value).strip() if value is not None else ""
                    if text:
                        chunk_bytes = len(text.encode("utf-8"))
                        if written + chunk_bytes > max_bytes:
                            remaining = max_bytes - written
                            if remaining > 0:
                                trimmed_bytes = text.encode("utf-8")[:remaining]
                                trimmed = trimmed_bytes.decode("utf-8", errors="ignore")
                                f.write(trimmed)
                                written += len(trimmed.encode("utf-8"))
                            print(f"Wrote {written} bytes to {output_path}")
                            return
                        f.write(text)
                        written += chunk_bytes
        print(f"Wrote {written} bytes to {output_path}")
