from __future__ import annotations

from pathlib import Path

from .methods.hybrid_method import HybridMethod


def _fmt_amount(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _item_price(item):
    if getattr(item, "item_total", None) is not None:
        return _fmt_amount(item.item_total)
    if getattr(item, "unit_price", None) is not None:
        return _fmt_amount(item.unit_price)
    return ""


def process_image(path: str) -> dict:
    """
    Adapter layer for old app.py contract.
    Returns the same shape expected by the original main app.
    """
    extractor = HybridMethod()
    receipt = extractor.extract(path, receipt_id=Path(path).stem)

    fields = receipt.fields

    items = []
    for item in receipt.items:
        name = getattr(item, "name", None) or "Unknown"
        price = _item_price(item)
        items.append((name, price))

    card_value = fields.card_last4 or ""

    return {
        "store": fields.merchant_name or "Unknown",
        "date": fields.date or "",
        "time": fields.time or "",
        "subtotal": _fmt_amount(fields.subtotal),
        "tax": _fmt_amount(fields.tax),
        "total": _fmt_amount(fields.total),
        "card": card_value,
        "payment_method": fields.payment_method or "",
        "receipt_number": fields.receipt_number or "",
        "items": items,
        "review_required": getattr(receipt, "review_required", False),
        "review_reasons": getattr(receipt, "review_reasons", []),
        "field_sources": getattr(receipt, "field_sources", {}),
        "item_source": getattr(receipt, "item_source", ""),
    }
