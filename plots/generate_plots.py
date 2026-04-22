import sys
import argparse
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plots.utils import (
    REPORTS_DIR, PREDICTIONS_DIR, IMAGES_DIR, ANNOTATIONS_DIR,
    METHOD_COLORS, METHOD_MAPPING, load_summary, load_per_receipt, get_store_from_id
)


# Ensure plots directory exists
PLOTS_OUT_DIR = Path("plots")
PLOTS_OUT_DIR.mkdir(exist_ok=True)

METHODS = ["ocr_regex", "ocr_layoutlm", "vlm", "hybrid"]
FIELDS = ["merchant_name", "date", "time", "subtotal", "tax", "total", "payment_method", "card_last4", "receipt_number"]

def plot_1_method_comparison():
    data = []
    for m in METHODS:
        summary = load_summary(m)
        if not summary: continue
        display_name = METHOD_MAPPING.get(m, m)
        data.append({"Method": display_name, "Metric": "Core Field Mean", "Score": summary.get("core_field_mean", 0)})
        data.append({"Method": display_name, "Metric": "Item Name F1", "Score": summary.get("item_name_f1", 0)})
        data.append({"Method": display_name, "Metric": "Overall Score", "Score": summary.get("overall_score", 0)})
    
    if not data:
        return
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Method", palette=METHOD_COLORS)
    plt.title("Overall Method Comparison")
    plt.ylim(0, 1.15)
    plt.ylabel("Accuracy / F1 Score")
    plt.legend(title="Method")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_1_method_comparison.png")
    plt.close()

def plot_2_field_heatmap():
    data = {}
    for m in METHODS:
        summary = load_summary(m)
        if not summary: continue
        display_name = METHOD_MAPPING.get(m, m)
        data[display_name] = [summary.get(f, 0) for f in FIELDS]
        
    if not data: return
    
    df = pd.DataFrame(data, index=FIELDS).T
    plt.figure(figsize=(12, 4))
    sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Field-wise Accuracy Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_2_field_heatmap.png")
    plt.close()

def plot_3_score_distribution():
    data = []
    for m in METHODS:
        receipts = load_per_receipt(m)
        display_name = METHOD_MAPPING.get(m, m)
        for r in receipts:
            score = r.get("metrics", {}).get("overall_score", 0)
            data.append({"Method": display_name, "Overall Score": score})
            
    if not data: return
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Overall Score", palette=METHOD_COLORS)
    plt.title("Per-receipt Overall Score Distribution")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_3_score_distribution.png")
    plt.close()

def plot_4_store_performance():
    data = []
    for m in METHODS:
        receipts = load_per_receipt(m)
        display_name = METHOD_MAPPING.get(m, m)
        for r in receipts:
            score = r.get("metrics", {}).get("overall_score", 0)
            store = get_store_from_id(r.get("receipt_id", ""))
            data.append({"Method": display_name, "Store": store, "Overall Score": score})
            
    if not data: return
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Store", y="Overall Score", hue="Method", palette=METHOD_COLORS, errorbar=None)
    plt.title("Store-wise Average Overall Score")
    plt.ylim(0, 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_4_store_performance.png")
    plt.close()

def plot_5_item_metrics():
    data = []
    for m in METHODS:
        summary = load_summary(m)
        if not summary: continue
        display_name = METHOD_MAPPING.get(m, m)
        data.append({"Method": display_name, "Metric": "Item Precision", "Score": summary.get("item_name_precision", 0)})
        data.append({"Method": display_name, "Metric": "Item Recall", "Score": summary.get("item_name_recall", 0)})
        data.append({"Method": display_name, "Metric": "Item F1", "Score": summary.get("item_name_f1", 0)})
    
    if not data: return
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Method", palette=METHOD_COLORS)
    plt.title("Item Extraction Performance")
    plt.ylim(0, 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_5_item_metrics.png")
    plt.close()

def _save_single_qualitative(rid: str, title: str, filename: str) -> None:
    """Save a single qualitative receipt example as a standalone figure."""
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = IMAGES_DIR / f"{rid}{ext}"
        if p.exists():
            img_path = p
            break

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))

    if not img_path:
        ax.text(0.5, 0.5, f"Image not found: {rid}", ha="center", va="center", fontsize=14)
        ax.set_title(f"{title} ({rid}) — IMAGE NOT FOUND")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(PLOTS_OUT_DIR / filename, bbox_inches="tight")
        plt.close()
        return

    img = Image.open(img_path).convert("RGB")
    try:
        from app.pipeline import run_ocr_layoutlm_pipeline
        proto_result = run_ocr_layoutlm_pipeline(str(img_path))
        draw = ImageDraw.Draw(img)
        for field in proto_result.fields:
            if field.bbox:
                draw.rectangle(field.bbox, outline="blue", width=3)
        for item in proto_result.items:
            if item.description and item.description.bbox:
                draw.rectangle(item.description.bbox, outline="purple", width=2)
            if item.price and item.price.bbox:
                draw.rectangle(item.price.bbox, outline="red", width=2)
    except Exception as e:
        print(f"  Note: Could not draw bboxes for {rid}: {e}")

    ax.imshow(img)
    ax.set_title(f"{title}\n({rid})", fontsize=14, fontweight="bold")
    ax.axis("off")

    hybrid_pred_path = PREDICTIONS_DIR / "hybrid" / f"{rid}.json"
    if hybrid_pred_path.exists():
        with open(hybrid_pred_path, "r") as f:
            h_pred = json.load(f)
        lines = ["Hybrid Extracted Fields:"]
        for k, v in h_pred.get("fields", {}).items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        lines.append(f"\nItems: {len(h_pred.get('items', []))}")
        ax.text(1.02, 0.5, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / filename, bbox_inches="tight", dpi=100)
    plt.close()


def plot_6_qualitative_examples(best_id=None, worst_id=None):
    hybrid_receipts = load_per_receipt("hybrid")
    if not hybrid_receipts:
        print("No hybrid per_receipt data found for Plot 6")
        return

    hybrid_receipts.sort(key=lambda x: x.get("metrics", {}).get("overall_score", 0))

    if not worst_id:
        worst_id = hybrid_receipts[0]["receipt_id"]
    if not best_id:
        best_id = hybrid_receipts[-1]["receipt_id"]

    _save_single_qualitative(worst_id, "Difficult Failure Case", "plot_6a_failure_case.png")
    _save_single_qualitative(best_id,  "Strong Success Case",   "plot_6b_success_case.png")

def plot_7_disagreement_reasons():
    hybrid_receipts = load_per_receipt("hybrid")
    if not hybrid_receipts: return
    
    reasons = []
    for r in hybrid_receipts:
        reasons.extend(r.get("review_reasons", []))
        
    if not reasons: return
    
    from collections import Counter
    counts = Counter(reasons)
    df = pd.DataFrame(counts.items(), columns=["Reason", "Count"]).sort_values("Count", ascending=True)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Count", y="Reason", color=METHOD_COLORS["Hybrid"])
    plt.title("Hybrid Routing: Disagreement Reasons (Triggers for Manual Review)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_7_disagreement_reasons.png")
    plt.close()

def plot_8_missing_field_rate():
    # A field is 'missing' or 'failed' if its score is 0.0
    hard_fields = ["payment_method", "card_last4", "receipt_number", "time"]
    data = []
    
    for m in METHODS:
        receipts = load_per_receipt(m)
        if not receipts: continue
        display_name = METHOD_MAPPING.get(m, m)
        
        total = len(receipts)
        if total == 0: continue
            
        for f in hard_fields:
            missing_count = sum(1 for r in receipts if r.get("metrics", {}).get(f) == 0.0)
            data.append({
                "Method": display_name,
                "Field": f,
                "Missing Rate (%)": (missing_count / total) * 100
            })
            
    if not data: return
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Field", y="Missing Rate (%)", hue="Method", palette=METHOD_COLORS)
    plt.title("Missing/Failed Field Rate by Method")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_8_missing_field_rate.png")
    plt.close()

def plot_9_confidence_distribution():
    print("Running Prototype on test dataset to extract confidences for Plot 9...")
    data = []
    # Just take up to 20 images to not make it run forever
    img_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpeg"))
    
    count = 0
    for img_path in img_files:
        try:
            from app.pipeline import run_ocr_layoutlm_pipeline
            res = run_ocr_layoutlm_pipeline(str(img_path))
            for f in res.fields:
                if f.confidence is not None:
                    data.append({"Field": f.field_name, "Confidence": f.confidence})
            count += 1
            if count >= 20: # Limit for speed, can be increased
                break
        except Exception:
            continue
            
    if not data:
        print("No confidence data collected for Plot 9")
        return
        
    df = pd.DataFrame(data)
    # Filter to main fields
    main_fields = ["merchant_name", "date", "subtotal", "tax", "total"]
    df = df[df["Field"].isin(main_fields)]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Field", y="Confidence", color=METHOD_COLORS["OCR+LayoutLM"], inner="quartile")
    plt.title("Prototype Field Confidence Distribution")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_9_confidence_distribution.png")
    plt.close()

def plot_10_annotation_correction():
    files = list(ANNOTATIONS_DIR.glob("*_annotation.json"))
    if not files:
        print("No annotation files found for Plot 10. Creating empty plot.")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No User Annotations Available Yet", ha='center', va='center', fontsize=15)
        plt.axis("off")
        plt.savefig(PLOTS_OUT_DIR / "plot_10_annotation_correction.png")
        plt.close()
        return
        
    corrected_count = 0
    deleted_count = 0
    added_boxes = 0
    
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
            
        for tok in data.get("tokens", []):
            if str(tok.get("status", "")).lower() == "deleted":
                deleted_count += 1
            elif "corrected_label" in tok and tok["corrected_label"] != tok.get("label"):
                corrected_count += 1
                
        added_boxes += len(data.get("manual_boxes", []))
        
    df = pd.DataFrame([
        {"Action": "Corrected Labels", "Count": corrected_count},
        {"Action": "Deleted Boxes", "Count": deleted_count},
        {"Action": "Added Manual Boxes", "Count": added_boxes}
    ])
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x="Action", y="Count", color="#9467bd")
    plt.title("Annotation Corrections Summary")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT_DIR / "plot_10_annotation_correction.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-receipt", type=str, help="Receipt ID for the 'best' qualitative example")
    parser.add_argument("--worst-receipt", type=str, help="Receipt ID for the 'worst' qualitative example")
    args = parser.parse_args()

    print("Generating Plot 1...")
    plot_1_method_comparison()
    print("Generating Plot 2...")
    plot_2_field_heatmap()
    print("Generating Plot 3...")
    plot_3_score_distribution()
    print("Generating Plot 4...")
    plot_4_store_performance()
    print("Generating Plot 5...")
    plot_5_item_metrics()
    print("Generating Plot 6...")
    plot_6_qualitative_examples(args.best_receipt, args.worst_receipt)
    print("Generating Plot 7...")
    plot_7_disagreement_reasons()
    print("Generating Plot 8...")
    plot_8_missing_field_rate()
    print("Generating Plot 9... [SKIPPED — keeping existing file]")
    # plot_9_confidence_distribution()  # slow; requires full OCR pipeline
    print("Generating Plot 10...")
    plot_10_annotation_correction()
    print("All plots generated in 'plots' folder!")
