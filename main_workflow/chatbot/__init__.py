import os
import re
from pathlib import Path

from .csv_loader import get_connection
from .schema_probe import probe_schema, build_context_string
from .sql_runner import run_sql
from .session_manager import SessionManager

_sessions = SessionManager()
_model = None
_ask_fn = None


def _lazy_query_engine():
    global _model, _ask_fn
    if _ask_fn is None:
        from .query_engine import get_model, ask
        _ask_fn = ask
        if _model is None:
            _model = get_model()
    return _ask_fn, _model


def _storage_mode() -> str:
    mode = os.environ.get("BILLWISE_STORAGE_MODE", "auto").strip().lower()
    if mode not in {"auto", "local", "gcs"}:
        mode = "auto"
    return mode


def _default_local_csv() -> str:
    env_path = os.environ.get("LOCAL_BILLS_CSV", "").strip()
    if env_path:
        return str(Path(env_path).resolve())
    return str(Path(__file__).resolve().parents[1] / "data" / "dev" / "bills_output.csv")


def _resolve_source(csv_source: str | None) -> tuple[str, str]:
    if csv_source:
        return csv_source, "local" if os.path.exists(csv_source) else "gcs"

    mode = _storage_mode()
    env_blob = os.environ.get("GCS_BILLS_BLOB", "bills_output.csv")
    bucket = os.environ.get("GCS_BUCKET_NAME", "").strip()

    if mode == "local":
        return _default_local_csv(), "local"

    if mode == "gcs":
        return env_blob, "gcs"

    if bucket:
        return env_blob, "gcs"

    return _default_local_csv(), "local"


def _ensure_loaded(session_id: str, csv_source: str | None):
    if _sessions.has_connection(session_id):
        return

    source, source_type = _resolve_source(csv_source)
    conn = get_connection(source, source_type)
    schema = build_context_string(probe_schema(conn))
    _sessions.set_connection(session_id, conn, schema)


def _simple_local_answer(conn, message: str) -> str | None:
    q = message.lower().strip()

    if any(p in q for p in [
        "how much did i spend",
        "total spend",
        "total spent",
        "how much have i spent",
    ]):
        sql = """
        SELECT COALESCE(SUM(bill_total), 0) AS total_spend
        FROM (
            SELECT Serial_No, MAX(TRY_CAST(Total AS DOUBLE)) AS bill_total
            FROM data
            GROUP BY Serial_No
        )
        """
        value = conn.execute(sql).fetchone()[0]
        value = 0.0 if value is None else float(value)
        return f"You spent ${value:.2f}."

    if any(p in q for p in ["how many bills", "how many receipts", "bill count", "receipt count"]):
        sql = "SELECT COUNT(DISTINCT Serial_No) FROM data"
        count = conn.execute(sql).fetchone()[0]
        return f"You have {int(count)} bill(s)."

    if any(p in q for p in ["average spend", "average bill", "avg bill", "avg spend"]):
        sql = """
        SELECT COALESCE(AVG(bill_total), 0) AS avg_spend
        FROM (
            SELECT Serial_No, MAX(TRY_CAST(Total AS DOUBLE)) AS bill_total
            FROM data
            GROUP BY Serial_No
        )
        """
        value = conn.execute(sql).fetchone()[0]
        value = 0.0 if value is None else float(value)
        return f"Your average bill is ${value:.2f}."

    if any(p in q for p in ["which stores", "list stores", "what stores"]):
        sql = """
        SELECT DISTINCT Store_Name
        FROM data
        WHERE Store_Name IS NOT NULL AND TRIM(Store_Name) <> ''
        ORDER BY Store_Name
        """
        rows = conn.execute(sql).fetchall()
        stores = [r[0] for r in rows if r[0]]
        if not stores:
            return "No store names are available yet."
        return "Stores found: " + ", ".join(stores)

    if any(p in q for p in ["latest bill", "last bill", "most recent bill"]):
        sql = """
        SELECT Serial_No, MAX(Store_Name) AS store_name, MAX(Invoice_Date) AS invoice_date, MAX(TRY_CAST(Total AS DOUBLE)) AS total
        FROM data
        GROUP BY Serial_No
        ORDER BY Serial_No DESC
        LIMIT 1
        """
        row = conn.execute(sql).fetchone()
        if not row:
            return "No bill data is available yet."
        serial_no, store_name, invoice_date, total = row
        total = 0.0 if total is None else float(total)
        return f"Latest bill: #{serial_no} from {store_name or 'Unknown'} on {invoice_date or 'unknown date'} for ${total:.2f}."

    return None


def handle_chat_message(
    session_id: str,
    message: str,
    csv_source: str | None = None,
) -> str:
    lower = message.strip().lower()

    if lower in ("reset", "clear", "start over"):
        _sessions.clear(session_id)
        return "Conversation reset. Ask me anything about your spending!"

    if lower in ("reload", "refresh", "refresh data"):
        _sessions.reload_csv(session_id)
        _ensure_loaded(session_id, csv_source)
        return "Data reloaded from storage. What would you like to know?"

    try:
        _ensure_loaded(session_id, csv_source)
    except Exception as e:
        return f"Could not load the data file: {e}"

    conn = _sessions.get_connection(session_id)
    schema = _sessions.get_schema(session_id)
    history = _sessions.get_history(session_id)

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        fallback = _simple_local_answer(conn, message)
        if fallback is not None:
            return fallback
        return (
            "GEMINI_API_KEY is not set. In local mode I can answer simple questions like:\n"
            "- How much did I spend?\n"
            "- How many bills do I have?\n"
            "- Which stores are in my data?\n"
            "- What is my average bill?\n"
            "- What is my latest bill?"
        )

    try:
        ask_fn, model = _lazy_query_engine()
        reply, sql = ask_fn(message, schema, history, model)
    except Exception as e:
        return f"I had trouble understanding that question: {e}"

    if sql is None:
        clean = re.sub(r"<sql>.*?</sql>", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
        return clean or reply

    success, result = run_sql(sql, conn)

    if not success:
        return (
            "I could not compute that - the generated query had an issue.\n"
            f"Details: {result}"
        )

    explanation = re.sub(r"<sql>.*?</sql>", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
    return f"{explanation}\n\n{result}" if explanation else result


def reload_session(session_id: str):
    _sessions.reload_csv(session_id)


def clear_session(session_id: str):
    _sessions.clear(session_id)


__all__ = ["handle_chat_message", "reload_session", "clear_session"]
