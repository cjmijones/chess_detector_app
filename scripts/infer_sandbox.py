import argparse
from pathlib import Path
from sandbox_loader import create_val_sandbox, cleanup_root
from predict import show_board

parser = argparse.ArgumentParser()
parser.add_argument("image", help="path to a jpg/png")
parser.add_argument("--ckpt", default="lightning_logs/version_6/checkpoints/"
                                    "best-epoch-epoch=11-val_loss=val_loss=0.2299.ckpt")
args = parser.parse_args()

img_bytes = Path(args.image).read_bytes()
model, val_set, idx, root = create_val_sandbox(img_bytes, args.ckpt)
show_board(model, val_set, idx)
cleanup_root(root)
