"""
sandbox_loader.py
-----------------
Build a temporary, *minimal* ChessReD data-root that contains exactly ONE
image (all 64 labels = 'empty').  Returns the model, the val-dataset and the
index of that new sample so you can call the usual `show_board()` or do your
own forward pass.

Usage:
    model, val_set, idx, tmp_dir = create_val_sandbox(
        img_bytes=...,                    # raw bytes from a JPEG/PNG
        chkpt_path="lightning_logs/...ckpt",
        px_resize=512
    )
"""
from __future__ import annotations
import json, tempfile, shutil, itertools, io
from pathlib import Path
from datetime import datetime

import torch
from torchvision.io import read_image

from chess_dataloader import ChessDataModule
from cnn_transformer_model import ChessCNNTransformer

# ──────────────────────────────────────────────────────────────────
# constants copied from your canonical annotations file once
FILES_RANKS = [f"{f}{r}" for r in "87654321" for f in "abcdefgh"]
CATEGORIES = [
    {"id": 0, "name": "empty"}, {"id": 1, "name": "w_pawn"},
    {"id": 2, "name": "w_knight"}, {"id": 3, "name": "w_bishop"},
    {"id": 4, "name": "w_rook"},  {"id": 5, "name": "w_queen"},
    {"id": 6, "name": "w_king"},  {"id": 7, "name": "b_pawn"},
    {"id": 8, "name": "b_knight"},{"id": 9, "name": "b_bishop"},
    {"id": 10,"name": "b_rook"},  {"id": 11,"name": "b_queen"},
    {"id": 12,"name": "b_king"},
]
EMPTY_ID = 0
# ──────────────────────────────────────────────────────────────────

def _make_empty_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="chess_sandbox_"))
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)

    stub = {
        "images": [],
        "annotations": {"pieces": [], "corners": []},
        "categories": CATEGORIES,
        "splits": {s: {"image_ids": [], "n_samples": 0} for s in ("train","val","test")}
    }
    (root / "annotations.json").write_text(json.dumps(stub, indent=2))
    return root


def _add_image(root: Path, img_bytes: bytes, split: str = "val") -> int:
    meta_path = root / "annotations.json"
    meta      = json.loads(meta_path.read_text())

    new_id = (max([img["id"] for img in meta["images"]] or [-1]) + 1)
    fname  = f"I{new_id:06d}.jpg"
    rel    = Path("images") / split / fname
    (root / rel).write_bytes(img_bytes)

    # -- images -------------------------------------------------------
    meta["images"].append({
        "id": new_id, "file_name": fname, "path": str(rel),
        "height": 0, "width": 0, "camera": "streamlit",
        "game_id": 0, "move_id": 0
    })

    # -- 64 'empty' annotations --------------------------------------
    next_ann_id = len(meta["annotations"]["pieces"])
    meta["annotations"]["pieces"].extend([
        {
            "id": next_ann_id + i,
            "image_id": new_id,
            "category_id": EMPTY_ID,
            "chessboard_position": pos,
        }
        for i, pos in enumerate(FILES_RANKS)
    ])

    # -- update splits ------------------------------------------------
    meta["splits"][split]["image_ids"].append(new_id)
    meta["splits"][split]["n_samples"] += 1

    meta_path.write_text(json.dumps(meta, indent=2))
    return new_id


def create_val_sandbox(
    img_bytes: bytes,
    chkpt_path: str,
    px_resize: int = 512,
):
    """Return model, val_set and idx for a *single* uploaded image."""
    root = _make_empty_root()
    img_id = _add_image(root, img_bytes, split="val")

    dm = ChessDataModule(str(root), batch_size=1, workers=0, px_resize=px_resize)
    dm.setup("fit")
    val_set = dm.chess_val
    idx = val_set.split_img_ids.index(img_id)

    model = ChessCNNTransformer.load_from_checkpoint(chkpt_path, map_location="cpu").eval()

    return model, val_set, idx, root  # caller may delete `root` afterwards


def cleanup_root(root: Path):
    shutil.rmtree(root, ignore_errors=True)
