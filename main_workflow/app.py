import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import logging
import hashlib
import tempfile

import requests
from flask import Flask, request, render_template, jsonify
from twilio.twiml.messaging_response import MessagingResponse

from main_workflow.extraction import process_image
from main_workflow.storage.csv_writer import append_bill, is_duplicate
from main_workflow.chatbot import handle_chat_message, reload_session, clear_session
from main_workflow import categorization as categorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main_workflow.app")

app = Flask(__name__)

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

_QUESTION_STARTERS = {
    "how", "what", "when", "show", "list", "which", "total",
    "sum", "average", "who", "where", "count", "give", "find",
    "reset", "clear", "reload", "refresh",
}

_categorizer_ready = False


def _ensure_categorizer() -> bool:
    global _categorizer_ready
    if _categorizer_ready:
        return True

    try:
        categorizer.init()
        _categorizer_ready = True
        log.info("Categorizer initialized successfully.")
        return True
    except Exception as exc:
        log.warning("Categorizer initialization failed. Continuing without categories. Error: %s", exc)
        return False


def _is_question(text: str) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    first_word = lower.split()[0] if lower else ""
    return lower.endswith("?") or first_word in _QUESTION_STARTERS


def _get_twilio_auth() -> tuple[str, str]:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set.")

    return account_sid, auth_token


def download_image(media_url: str) -> tuple[str, str]:
    """
    Download image from Twilio.
    Returns (temp_file_path, md5_hash).
    """
    account_sid, auth_token = _get_twilio_auth()

    resp = requests.get(media_url, auth=(account_sid, auth_token), timeout=30)
    resp.raise_for_status()

    image_bytes = resp.content
    image_hash = hashlib.md5(image_bytes).hexdigest()

    content_type = resp.headers.get("Content-Type", "image/jpeg")
    ext = "." + content_type.split("/")[-1].split(";")[0].strip()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(image_bytes)
    tmp.close()

    return tmp.name, image_hash


@app.route("/webhook", methods=["POST"])
def webhook():
    num_media = int(request.form.get("NumMedia", 0))
    sender = request.form.get("From", "unknown")
    resp = MessagingResponse()

    log.info("Incoming SMS from %s | media=%d", sender, num_media)

    if num_media == 0:
        text = request.form.get("Body", "").strip()
        log.info("Text message: %r", text)

        if _is_question(text):
            answer = handle_chat_message(session_id=sender, message=text)
            resp.message(answer)
        else:
            resp.message(
                "Hi! Send me a bill image to digitize.\n"
                "Or ask a question about your spending, for example:\n"
                "\"How much did we spend last month?\""
            )
        return str(resp)

    reply_parts = []

    for i in range(num_media):
        media_url = request.form.get(f"MediaUrl{i}")
        media_type = request.form.get(f"MediaContentType{i}", "")

        if media_type not in SUPPORTED_TYPES:
            reply_parts.append(
                f"Attachment {i + 1}: unsupported type ({media_type}). Send JPG, PNG, or WEBP."
            )
            continue

        img_path = None

        try:
            log.info("[%d/%d] Downloading image...", i + 1, num_media)
            img_path, image_hash = download_image(media_url)

            log.info("[%d/%d] Dedup Layer 1: hash check", i + 1, num_media)
            dupe, match = is_duplicate(image_hash, "", "", "")
            if dupe:
                reply_parts.append(
                    f"Bill already submitted.\n"
                    f"Original entry: Bill #{match[0]} | {match[2]} | {match[3]} | ${match[4]}"
                )
                continue

            log.info("[%d/%d] Running hybrid extraction...", i + 1, num_media)
            result = process_image(img_path)

            log.info(
                "[%d/%d] Extraction result - store=%r | date=%r | total=%r | card=%r | items=%d",
                i + 1,
                num_media,
                result["store"],
                result["date"],
                result["total"],
                result["card"],
                len(result["items"]),
            )

            log.info("[%d/%d] Dedup Layer 2: content check", i + 1, num_media)
            dupe, match = is_duplicate(
                image_hash=image_hash,
                store=result["store"],
                date=result["date"],
                total=result["total"],
            )
            if dupe:
                reply_parts.append(
                    f"Bill already submitted.\n"
                    f"Original entry: Bill #{match[0]} | {match[2]} | {match[3]} | ${match[4]}"
                )
                continue

            categorizer_ready = _ensure_categorizer()
            categorized_items = []

            for name, price in result["items"]:
                category = ""
                if categorizer_ready and name:
                    try:
                        category = categorizer.categorize(name)
                    except Exception as exc:
                        log.warning("Categorization failed for item %r: %s", name, exc)
                        category = ""
                categorized_items.append((name, price, category))

            log.info("[%d/%d] Writing extracted bill to cloud storage...", i + 1, num_media)
            serial = append_bill(
                filename=os.path.basename(img_path),
                store=result["store"],
                date=result["date"],
                total=result["total"],
                card=result["card"],
                sender=sender,
                image_hash=image_hash,
                items=categorized_items,
            )

            reload_session(sender)

            saved_message = (
                f"Bill #{serial} saved\n"
                f"Store : {result['store']}\n"
                f"Date  : {result['date'] or 'Not found'}\n"
                f"Card  : {result['card'] or 'Not found'}\n"
                f"Total : {result['total'] or 'Not found'}"
            )

            if result.get("review_required"):
                reasons = result.get("review_reasons") or []
                if reasons:
                    saved_message += "\nReview suggested: " + "; ".join(map(str, reasons))
                else:
                    saved_message += "\nReview suggested: low-confidence extraction."

            reply_parts.append(saved_message)

        except Exception as exc:
            log.exception("Error processing image %d: %s", i + 1, exc)
            reply_parts.append(f"Could not process image {i + 1}: {exc}")

        finally:
            if img_path and os.path.exists(img_path):
                os.unlink(img_path)

    resp.message("\n\n".join(reply_parts))
    return str(resp)


@app.route("/chat")
def chat_ui():
    return render_template("chat.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    question = data.get("question", "").strip()
    csv_source = data.get("csv_source")

    if not question:
        return jsonify({"success": False, "error": "No question provided."}), 400

    try:
        answer = handle_chat_message(
            session_id=session_id,
            message=question,
            csv_source=csv_source,
        )
        return jsonify({"success": True, "answer": answer})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    clear_session(session_id)
    return jsonify({"success": True, "message": "Session cleared."})


@app.route("/api/reload", methods=["POST"])
def api_reload():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    reload_session(session_id)
    return jsonify({"success": True, "message": "Data will be reloaded on next query."})


@app.route("/", methods=["GET"])
def health():
    return "BillWise main workflow is running", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
