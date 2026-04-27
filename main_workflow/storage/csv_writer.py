import csv
import io
import os
import logging
import threading
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

HEADERS = [
    "Serial_No",
    "Bill_File",
    "Store_Name",
    "Invoice_Date",
    "Total",
    "Card_Used",
    "Received_At",
    "Sender",
    "Image_Hash",
    "Item_Name",
    "Item_Price",
    "Grocery_Category",
]

STORE_SIMILARITY_THRESHOLD = 0.85

_lock = threading.Lock()
_gcs_client = None


def _main_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_local_csv() -> Path:
    return _main_root() / "data" / "dev" / "bills_output.csv"


def _get_local_csv_path() -> Path:
    raw = os.environ.get("LOCAL_BILLS_CSV", "").strip()
    path = Path(raw) if raw else _default_local_csv()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _storage_mode() -> str:
    mode = os.environ.get("BILLWISE_STORAGE_MODE", "auto").strip().lower()
    if mode not in {"auto", "local", "gcs"}:
        mode = "auto"
    return mode


def _use_gcs() -> bool:
    mode = _storage_mode()
    if mode == "local":
        return False
    if mode == "gcs":
        return True
    return bool(os.environ.get("GCS_BUCKET_NAME", "").strip())


def _get_bucket_name() -> str:
    bucket = os.environ.get("GCS_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("GCS_BUCKET_NAME is not set.")
    return bucket


def _get_blob_name() -> str:
    return os.environ.get("GCS_BILLS_BLOB", "bills_output.csv")


def _get_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _get_bucket():
    return _get_client().bucket(_get_bucket_name())


def _read_rows_local() -> list[list]:
    path = _get_local_csv_path()
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    return rows[1:] if rows else []


def _write_rows_local(rows: list[list]):
    path = _get_local_csv_path()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def _read_rows_gcs() -> list[list]:
    bucket = _get_bucket()
    blob = bucket.blob(_get_blob_name())

    if not blob.exists():
        return []

    content = blob.download_as_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    return rows[1:] if rows else []


def _write_rows_gcs(rows: list[list]):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(HEADERS)
    writer.writerows(rows)

    bucket = _get_bucket()
    blob = bucket.blob(_get_blob_name())
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")


def _read_rows() -> list[list]:
    if _use_gcs():
        return _read_rows_gcs()
    return _read_rows_local()


def _write_rows(rows: list[list]):
    if _use_gcs():
        _write_rows_gcs(rows)
    else:
        _write_rows_local(rows)


def _fuzzy_score(s1: str, s2: str) -> float:
    s1, s2 = s1.upper().strip(), s2.upper().strip()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    return (2.0 * lcs_len) / (m + n)


def is_duplicate(image_hash: str, store: str, date: str, total: str):
    rows = _read_rows()

    for row in rows:
        if len(row) < 9:
            continue

        existing_hash = row[8].strip()
        existing_store = row[2].strip()
        existing_date = row[3].strip()
        existing_total = row[4].strip()

        if image_hash and existing_hash == image_hash:
            return True, row

        date_match = bool(date and existing_date == date)
        total_match = bool(total and existing_total == str(total))
        store_score = _fuzzy_score(store, existing_store)
        store_match = store_score >= STORE_SIMILARITY_THRESHOLD

        if date_match and total_match and store_match:
            return True, row

    return False, None


def append_bill(filename, store, date, total, card, sender, image_hash, items) -> int:
    if not items:
        items = [("Unknown", "", "")]

    with _lock:
        existing = _read_rows()

        if existing:
            try:
                serial = max(int(row[0]) for row in existing if row) + 1
            except (ValueError, IndexError):
                serial = len(existing) + 1
        else:
            serial = 1

        received = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        for item_name, item_price, item_category in items:
            new_row = [
                serial,
                filename,
                store or "Not found",
                date or "Not found",
                total or "Not found",
                card or "",
                received,
                sender,
                image_hash,
                item_name,
                item_price,
                item_category,
            ]
            existing.append(new_row)

        _write_rows(existing)

        backend = "GCS" if _use_gcs() else "local CSV"
        log.info(
            "Bill #%d written to %s - store=%r | date=%s | total=%s | %d item rows",
            serial,
            backend,
            store,
            date,
            total,
            len(items),
        )

    return serial


def reset_csv():
    with _lock:
        _write_rows([])
