"""
smoke_test_ocr_regex.py — Quick sanity check for the OCR+Regex method.

Runs GoogleVisionRegexMethod on the first 2 receipts that have a matching
gold label and prints the extracted fields vs. ground truth.

Usage (from repo root):
    python scripts/smoke_test_ocr_regex.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Make sure repo root is on sys.path when running directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.loaders import load_gold_receipt
from evaluation.metrics import score_receipt
from methods.ocr_regex import GoogleVisionRegexMethod


REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_DIR  = REPO_ROOT / "assets" / "dataset"
RECEIPTS_DIR = DATASET_DIR / "receipts"
GOLD_DIR     = DATASET_DIR / "gold_labels"

MAX_RECEIPTS = 2   # change to test more


def _banner(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def main() -> None:
    method = GoogleVisionRegexMethod()

    gold_files = sorted(GOLD_DIR.glob("*.json"))
    if not gold_files:
        print(f"[ERROR] No gold label files found in {GOLD_DIR}")
        sys.exit(1)

    tested = 0
    for gold_path in gold_files:
        if tested >= MAX_RECEIPTS:
            break

        gold = load_gold_receipt(gold_path)
        receipt_id = gold.receipt_id
        image_path = RECEIPTS_DIR / gold.image_file

        if not image_path.exists():
            print(f"[SKIP] {receipt_id} — image not found: {image_path}")
            continue

        _banner(f"Receipt: {receipt_id}  |  Image: {gold.image_file}")

        try:
            pred = method.extract(str(image_path), receipt_id)
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            tested += 1
            continue

        # Field comparison table
        print(f"\n{'Field':<18}  {'Gold':^25}  {'Predicted':^25}")
        print(f"{'─'*18}  {'─'*25}  {'─'*25}")

        gold_fields = gold.fields.model_dump()
        pred_fields = pred.fields.model_dump()
        for field, gold_val in gold_fields.items():
            pred_val = pred_fields.get(field)
            match_marker = "✓" if str(gold_val) == str(pred_val) else " "
            print(f"{match_marker} {field:<17}  {str(gold_val):<25}  {str(pred_val):<25}")

        # Item comparison
        print(f"\n  Gold items  ({len(gold.items)}): {[i.name for i in gold.items]}")
        print(f"  Pred items  ({len(pred.items)}): {[i.name for i in pred.items]}")

        # Score
        metrics = score_receipt(pred, gold)
        print(f"\n  core_field_mean : {metrics['core_field_mean']:.3f}")
        print(f"  item_name_f1    : {metrics['item_name_f1']:.3f}")
        print(f"  overall_score   : {metrics['overall_score']:.3f}")

        tested += 1

    if tested == 0:
        print("No receipts could be tested.")
    else:
        print(f"\n[Done] Smoke-tested {tested} receipt(s).")


if __name__ == "__main__":
    main()
