from predict import show_board
from chess_dataloader import ChessDataModule
from cnn_transformer_model import ChessCNNTransformer
from pathlib import Path
import torch, json

# -- load model -----------------------------
ckpt = Path("lightning_logs/version_6/checkpoints/"
            "best-epoch-epoch=11-val_loss=val_loss=0.2299.ckpt")
model = ChessCNNTransformer.load_from_checkpoint(str(ckpt), map_location="cpu").eval()

# -- build validation dataset ---------------
dm = ChessDataModule(dataroot="./extracted_data",
                     batch_size=1,
                     workers=0,
                     px_resize=512)
dm.setup("fit")
val_set = dm.chess_val

# -- look up index for id 389 ---------------
idx = val_set.split_img_ids.index(389)

# -- show the board -------------------------
show_board(model, val_set, idx, device="cpu")   # or "cuda" if available
