from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from google.cloud import vision

from evaluation.canonical import CanonicalFields, CanonicalLineItem, CanonicalReceipt
from evaluation.normalize import normalize_amount, normalize_payment_method
from methods.base import BaseExtractionMethod

log = logging.getLogger("ocr_regex")


# ---------------------------------------------------------------------------
# Module-level extraction helpers (improved from ocr_pipeline.py)
# ---------------------------------------------------------------------------

def _extract_text(client, path: str) -> str:
    with open(path, "rb") as img:
        content = img.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""


def _extract_date(text: str) -> str:
    text = text.replace("|", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text)

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2}\b",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{1,2}\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*/\d{1,2}/\d{4}\b",
    ]

    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append(m.group())

    for raw in found:
        raw_clean = raw.strip()
        for fmt in [
            "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
            "%m-%d-%Y", "%m-%d-%y", "%b %d %Y",
            "%B %d %Y", "%d %b %Y", "%d %B %Y",
            "%b/%d/%Y", "%B/%d/%Y",
        ]:
            try:
                dt = datetime.strptime(raw_clean.title(), fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return ""


def _extract_time(text: str) -> str:
    patterns = [
        r"\b\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)\b",
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b",
        r"\b\d{1,2}:\d{2}:\d{2}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group().strip()
    return ""


def _extract_subtotal(text: str):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    for line in lines:
        if "SUBTOTAL" in line or "SUB TOTAL" in line or "SUB-TOTAL" in line:
            m = re.search(money_pattern, line)
            if m:
                try:
                    return round(float(m.group().replace(",", "")), 2)
                except ValueError:
                    pass
    return None


def _extract_tax(text: str):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    for line in lines:
        if re.search(r"\bTAX\b", line) and "SUBTOTAL" not in line and "TOTAL" not in line:
            m = re.search(money_pattern, line)
            if m:
                try:
                    return round(float(m.group().replace(",", "")), 2)
                except ValueError:
                    pass
    return None


def _extract_payment_method_raw(text: str) -> str:
    """Returns a raw keyword string; will be normalized by normalize_payment_method()."""
    upper = text.upper()
    for keyword in ["VISA", "MASTERCARD", "MASTER CARD", "AMEX",
                    "AMERICAN EXPRESS", "DISCOVER", "DEBIT", "CREDIT", "CASH"]:
        if keyword in upper:
            return keyword
    return ""


def _extract_receipt_number(text: str) -> str:
    patterns = [
        r"(?:TRANS(?:ACTION)?|RECEIPT|ORDER|CHECK|TICKET)\s*(?:ID|NO|NUMBER|#)?\s*[:\-]?\s*([A-Z0-9\-]{4,})",
        r"(?:REF(?:ERENCE)?)\s*(?:ID|NO|NUMBER|#)?\s*[:\-]?\s*([A-Z0-9\-]{4,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _extract_card_last4(text: str) -> str:
    """Returns the 4-digit card suffix, or 'cash' if no card is found."""
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    card_keywords = ["VISA", "MASTERCARD", "CARD", "DEBIT", "CREDIT", "REFERENCE#"]

    for i, line in enumerate(lines):
        # Masked card pattern anywhere: XXXX1234 or ****1234
        match = re.search(r"[X\*]{4,}\d{4}", line)
        if match:
            return match.group()[-4:]

        if any(k in line for k in card_keywords):
            # 4 digits on the same line
            match = re.search(r"\b(\d{4})\b", line)
            if match:
                return match.group(1)
            # Card number may be on the next line (e.g. "Card No.\n8352")
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r"\b(\d{4})\b", next_line)
                if match:
                    return match.group(1)
    return "cash"


def _detect_store(lines) -> str:
    for line in lines[:15]:
        clean = line.strip()
        if not clean:
            continue
        if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", clean):
            continue
        if re.search(r"\d+\.\d{2}", clean):
            continue
        # Skip noise lines where less than half the chars are alphanumeric/space
        alnum = sum(c.isalnum() or c == " " for c in clean)
        if alnum / len(clean) < 0.5:
            continue
        if len(clean) > 3:
            return clean
    return "UNKNOWN"


def _extract_items(text: str) -> list[tuple[str, str]]:
    """
    Returns a list of (item_name, item_price_str) tuples.
    Handles both single-line and multi-line receipt formats.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    skip_keywords = [
        "TOTAL", "SUBTOTAL", "TAX", "BALANCE", "CHANGE", "CASH",
        "CREDIT", "DEBIT", "VISA", "MASTERCARD", "CARD", "DUE",
        "AMOUNT", "PAYMENT", "THANK", "SAVE", "DISCOUNT", "COUPON",
        "MEMBER", "REWARD", "POINT", "RECEIPT", "STORE", "TEL",
        "ADDRESS", "PHONE", "WWW", "HTTP", "APPROVED", "AUTH",
        "CASHIER", "REGISTER", "SAVED", "SAVING", "PRICE",
        "ITEM NAME", "QTY", "TRAN", "TRANS", "ID#", "REF#",
    ]

    money_pattern = r"\d[\d,]*\.\d{2}"

    def _is_skip(line: str) -> bool:
        upper = line.upper()
        return any(k in upper for k in skip_keywords)

    def _is_name_line(line: str) -> bool:
        if re.search(money_pattern, line):
            return False
        if _is_skip(line):
            return False
        if not re.search(r"[A-Za-z]{3,}", line):
            return False
        alnum = sum(c.isalnum() or c == " " for c in line)
        if alnum / max(len(line), 1) < 0.4:
            return False
        name = re.sub(r"^\d+\s+", "", line).strip()
        return len(name) >= 3

    def _clean_name(line: str) -> str:
        name = re.sub(money_pattern, "", line)
        name = re.sub(r"^\d+\s+", "", name)      # strip leading item number
        name = re.sub(r"\s+[NS]\s*$", "", name)  # strip trailing tax flag
        return name.strip(" .-@#*/\\")

    items = []
    for i, line in enumerate(lines):
        price_match = re.search(money_pattern, line)
        if not price_match:
            continue
        if _is_skip(line):
            continue

        all_prices = re.findall(money_pattern, line)
        price = all_prices[-1]

        name = _clean_name(line)

        # If name is too short, try the previous line
        if len(name) < 3 and i > 0:
            prev = lines[i - 1]
            if _is_name_line(prev):
                name = re.sub(r"^\d+\s+", "", prev).strip()

        if len(name) < 2:
            continue

        items.append((name, price))

    return items


def _extract_total(text: str):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    amounts = []
    keyword_amounts = []

    for line in lines:
        clean_line = line.replace(",", "")
        matches = re.findall(money_pattern, clean_line)
        for m in matches:
            value = float(m)
            if 0 < value < 20000:
                amounts.append(value)
                if any(k in line for k in ["TOTAL", "AMOUNT", "BALANCE", "DUE", "PURCHASE"]):
                    keyword_amounts.append(value)

    if keyword_amounts:
        return max(keyword_amounts)
    if amounts:
        return max(amounts)
    return None


def _process_image(client, path: str) -> dict:
    """Run the full OCR+Regex pipeline on a single image. Returns a raw extraction dict."""
    log.info("Calling Google Cloud Vision OCR on %s", path)
    text = _extract_text(client, path)
    log.info("OCR returned %d characters", len(text))

    lines = text.split("\n")
    store          = _detect_store(lines)
    date           = _extract_date(text)
    time           = _extract_time(text)
    subtotal       = _extract_subtotal(text)
    tax            = _extract_tax(text)
    total          = _extract_total(text)
    card           = _extract_card_last4(text)
    payment_method = _extract_payment_method_raw(text)
    receipt_number = _extract_receipt_number(text)
    items          = _extract_items(text)

    log.info(
        "Extracted — store=%r | date=%r | total=%r | card=%r | items=%d",
        store, date, total, card, len(items),
    )

    return {
        "store":          store,
        "date":           date or None,
        "time":           time or None,
        "subtotal":       subtotal,
        "tax":            tax,
        "total":          total,
        "card":           card,
        "payment_method": payment_method or None,
        "receipt_number": receipt_number or None,
        "items":          items,   # list of (name, price_str) tuples
    }


# ---------------------------------------------------------------------------
# Method class
# ---------------------------------------------------------------------------

class GoogleVisionRegexMethod(BaseExtractionMethod):
    name = "ocr_regex"

    def __init__(self) -> None:
        self.client = vision.ImageAnnotatorClient()

    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        raw = _process_image(self.client, image_path)

        card_raw = raw.get("card", "cash")
        pm_raw   = raw.get("payment_method") or ""

        # Resolve card_last4 and payment_method
        payment_method = None
        card_last4     = None

        if isinstance(card_raw, str) and card_raw.lower() == "cash":
            payment_method = normalize_payment_method("cash")
        elif isinstance(card_raw, str) and re.fullmatch(r"\d{4}", card_raw):
            # A card digit was found — use the keyword-detected pm if available
            payment_method = normalize_payment_method(pm_raw or "card")
            card_last4 = card_raw
        elif pm_raw:
            payment_method = normalize_payment_method(pm_raw)

        # Build line items from (name, price_str) tuples
        items = [
            CanonicalLineItem(
                line_id=i + 1,
                name=name,
                quantity=None,
                unit_price=None,
                item_total=normalize_amount(price),
            )
            for i, (name, price) in enumerate(raw.get("items", []))
        ]

        return CanonicalReceipt(
            receipt_id=receipt_id,
            image_file=Path(image_path).name,
            fields=CanonicalFields(
                merchant_name=raw.get("store") or None,
                date=raw.get("date"),
                time=raw.get("time"),
                subtotal=normalize_amount(raw.get("subtotal")),
                tax=normalize_amount(raw.get("tax")),
                total=normalize_amount(raw.get("total")),
                payment_method=payment_method,
                card_last4=card_last4,
                receipt_number=raw.get("receipt_number"),
            ),
            items=items,
        )
