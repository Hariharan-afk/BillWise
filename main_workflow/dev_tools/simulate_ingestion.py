import os
import json
import hashlib
import argparse
from pathlib import Path

import main_workflow  # loads .env

from main_workflow.extraction import process_image
from main_workflow import categorization as categorizer
from main_workflow.storage.csv_writer import append_bill, is_duplicate
from main_workflow.chatbot import reload_session


def simulate_local_ingestion(
    image_path: str,
    sender: str = "local-sim",
    force_save: bool = False,
) -> dict:
    p = Path(image_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    image_bytes = p.read_bytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()

    # Layer 1 duplicate check: raw image hash
    dupe1, match1 = is_duplicate(image_hash, "", "", "")
    if dupe1 and not force_save:
        return {
            "status": "duplicate_layer1",
            "image_path": str(p),
            "image_hash": image_hash,
            "matched_row": match1,
        }

    # Extraction
    result = process_image(str(p))

    # Categorization
    categorized_items = []
    for name, price in result.get("items", []):
        category = ""
        if name:
            try:
                category = categorizer.categorize(name)
            except Exception as exc:
                category = ""
                print(f"[WARN] categorization failed for {name!r}: {exc}")
        categorized_items.append((name, price, category))

    # Layer 2 duplicate check: semantic/content duplicate
    dupe2, match2 = is_duplicate(
        image_hash=image_hash,
        store=result.get("store", ""),
        date=result.get("date", ""),
        total=result.get("total", ""),
    )
    if dupe2 and not force_save:
        return {
            "status": "duplicate_layer2",
            "image_path": str(p),
            "image_hash": image_hash,
            "extraction_result": result,
            "categorized_items": categorized_items,
            "matched_row": match2,
        }

    serial = append_bill(
        filename=os.path.basename(str(p)),
        store=result.get("store", ""),
        date=result.get("date", ""),
        total=result.get("total", ""),
        card=result.get("card", ""),
        sender=sender,
        image_hash=image_hash,
        items=categorized_items,
    )

    reload_session(sender)

    return {
        "status": "saved",
        "serial": serial,
        "image_path": str(p),
        "image_hash": image_hash,
        "extraction_result": result,
        "categorized_items": categorized_items,
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate BillWise ingestion using a local receipt image.")
    parser.add_argument("image_path", help="Path to local receipt image")
    parser.add_argument("--sender", default="local-sim", help="Logical sender/session id")
    parser.add_argument("--force-save", action="store_true", help="Save even if duplicate is detected")
    args = parser.parse_args()

    output = simulate_local_ingestion(
        image_path=args.image_path,
        sender=args.sender,
        force_save=args.force_save,
    )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
