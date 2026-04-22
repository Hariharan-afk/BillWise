"""
evaluate_ocr_layoutlm.py — Full evaluation of the OCR+LayoutLM (Prototype) method.

Usage (from repo root):
    python scripts/evaluate_ocr_layoutlm.py

Writes to:
    assets/dataset/evaluation_reports/ocr_layoutlm_per_receipt.json
    assets/dataset/evaluation_reports/ocr_layoutlm_summary.json
    assets/dataset/predictions/ocr_layoutlm/<receipt_id>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.loaders import load_gold_receipt
from evaluation.metrics import score_receipt, summarize_scores
from methods.ocr_layoutlm import PrototypeMethod


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    dataset_dir  = repo_root / "assets" / "dataset"
    receipts_dir = dataset_dir / "receipts"
    gold_dir     = dataset_dir / "gold_labels"

    predictions_root = dataset_dir / "predictions"
    reports_root     = dataset_dir / "evaluation_reports"

    method = PrototypeMethod()

    receipt_paths = sorted(
        p for p in receipts_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    print(f"\n=== Evaluating {method.name} ===")

    rows = []
    method_pred_dir = predictions_root / method.name
    method_pred_dir.mkdir(parents=True, exist_ok=True)

    for image_path in receipt_paths:
        receipt_id = image_path.stem
        gold_path  = gold_dir / f"{receipt_id}.json"

        if not gold_path.exists():
            print(f"  [SKIP] {receipt_id} — no gold label")
            continue

        gold = load_gold_receipt(gold_path)

        try:
            pred    = method.extract(str(image_path), receipt_id)
            metrics = score_receipt(pred, gold)

            save_json(method_pred_dir / f"{receipt_id}.json", pred.model_dump())

            rows.append({
                "receipt_id": receipt_id,
                "image_file": image_path.name,
                "metrics":    metrics,
            })

            print(
                f"  {receipt_id}: overall={metrics['overall_score']:.4f}  "
                f"core={metrics['core_field_mean']:.4f}  "
                f"items_f1={metrics['item_name_f1']:.4f}"
            )

        except Exception as e:
            rows.append({
                "receipt_id": receipt_id,
                "image_file": image_path.name,
                "error":      str(e),
            })
            print(f"  {receipt_id}: ERROR -> {e}")

    summary = summarize_scores(rows)

    save_json(reports_root / f"{method.name}_per_receipt.json", rows)
    save_json(reports_root / f"{method.name}_summary.json",     summary)

    print(f"\nSummary for {method.name}")
    for k, v in summary.items():
        print(f"  {k:<30} {v:.4f}")

    print(f"\nSaved reports to {reports_root}")


if __name__ == "__main__":
    main()
