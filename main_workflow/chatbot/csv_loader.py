from __future__ import annotations

import os
import tempfile
from pathlib import Path

import duckdb
import pandas as pd


def _load_csv(conn: duckdb.DuckDBPyConnection, csv_path: str) -> duckdb.DuckDBPyConnection:
    csv_path = Path(csv_path).resolve()

    df = pd.read_csv(csv_path, encoding="utf-8")
    conn.execute("DROP TABLE IF EXISTS data")
    conn.register("data_df", df)
    conn.execute("CREATE TABLE data AS SELECT * FROM data_df")
    conn.unregister("data_df")
    return conn


def _download_gcs_blob(blob_name: str) -> str:
    from google.cloud import storage

    bucket_name = os.environ.get("GCS_BUCKET_NAME", "").strip()
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is not set for GCS mode.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: gs://{bucket_name}/{blob_name}")

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    blob.download_to_filename(temp_path)
    return temp_path


def get_connection(source: str, source_type: str = "local") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")

    if source_type == "local":
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local CSV not found: {source}")
        return _load_csv(conn, source)

    if source_type == "gcs":
        temp_csv = _download_gcs_blob(source)
        return _load_csv(conn, temp_csv)

    raise ValueError(f"Unsupported source_type: {source_type}")
