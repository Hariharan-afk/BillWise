from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict

from . import classifier as clf
from . import abbreviation_normalization as norm

log = logging.getLogger(__name__)

_STATE: Dict[str, Any] = {
    "ready": False,
    "model": None,
    "tokenizer": None,
    "device": None,
    "inventory": None,
    "vectorizer": None,
    "tfidf_matrix": None,
    "config": None,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _main_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_config() -> dict:
    repo_root = _repo_root()
    main_root = _main_root()

    cfg = dict(clf.CONFIG)

    cfg["model_checkpoint"] = os.environ.get(
        "CATEGORIZER_MODEL_PATH",
        str(repo_root / "checkpoints" / "full_ft_distilbert_unweighted_best.pt"),
    )

    cfg["dataset_path"] = os.environ.get(
        "CATEGORIZER_DATASET_PATH",
        str(repo_root / "data" / "Processed_Datasets" / "Labeled" / "merged_labeled.csv"),
    )

    logs_dir = main_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg["unresolved_log"] = str(logs_dir / "unresolved_items.json")
    cfg["low_confidence_log"] = str(logs_dir / "low_confidence_items.json")
    cfg["human_review_log"] = str(logs_dir / "human_review_items.json")

    if os.environ.get("GEMINI_API_KEY"):
        cfg["gemini_api_key"] = os.environ["GEMINI_API_KEY"]

    return cfg


def init(force: bool = False) -> None:
    if _STATE["ready"] and not force:
        return

    cfg = _build_config()

    inventory, vectorizer, tfidf_matrix = norm.init_pipeline(cfg["dataset_path"])
    model, tokenizer, device = clf.load_classifier(cfg)

    _STATE.update(
        {
            "ready": True,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "inventory": inventory,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "config": cfg,
        }
    )

    log.info("Categorizer initialized successfully.")


def categorize(item_name: str) -> str:
    if not item_name:
        return ""

    if not _STATE["ready"]:
        init()

    result = clf.run_inference(
        item_text=item_name,
        model=_STATE["model"],
        tokenizer=_STATE["tokenizer"],
        device=_STATE["device"],
        inventory=_STATE["inventory"],
        vectorizer=_STATE["vectorizer"],
        tfidf_matrix=_STATE["tfidf_matrix"],
        config=_STATE["config"],
    )

    if result.get("needs_human_review") or result.get("flagged_unresolved"):
        return clf.HUMAN_REVIEW_NEEDED

    return result.get("final_label") or result.get("predicted_label") or ""


__all__ = ["init", "categorize"]
