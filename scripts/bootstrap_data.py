import os
from pathlib import Path


def _download_prefix(bucket, source_prefix: str, destination: Path) -> int:
    count = 0
    prefix = source_prefix.strip("/")
    if prefix:
        prefix = f"{prefix}/"

    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue

        relative = blob.name[len(prefix):] if prefix else blob.name
        if not relative:
            continue

        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)
        count += 1

    return count


def main() -> None:
    bucket_name = os.environ.get("TACTIQ_DATA_BUCKET")
    if not bucket_name:
        print("TACTIQ_DATA_BUCKET is not set; using local files only.")
        return

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    root = Path(__file__).resolve().parents[1]

    sync_paths = {
        os.environ.get("TACTIQ_RAW_DATA_PREFIX", "raw_data"): root / "raw_data",
        os.environ.get("TACTIQ_ASSETS_PREFIX", "assets"): root / "assets",
        os.environ.get("TACTIQ_GOZTEPE_ASSETS_PREFIX", "göztepehub/assets"): root / "göztepehub" / "assets",
        os.environ.get("TACTIQ_MODEL_PREFIX", "utils"): root / "utils",
    }

    total = 0
    for source_prefix, destination in sync_paths.items():
        if not source_prefix:
            continue
        downloaded = _download_prefix(bucket, source_prefix, destination)
        print(f"Downloaded {downloaded} file(s) from gs://{bucket_name}/{source_prefix}/")
        total += downloaded

    print(f"Cloud data bootstrap complete. Downloaded {total} file(s).")


if __name__ == "__main__":
    main()
