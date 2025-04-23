# scripts/predict.py
import argparse, random, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

from cnn_transformer_model import ChessCNNTransformer
from chess_dataloader import ChessDataModule


# ------------------------------------------------------------------ #
#  1. Inference on a slice of the validation set
# ------------------------------------------------------------------ #

def predict_range(model, dataset, start: int, end: int, device="cpu"):
    """
    Run the model on dataset[start:end] (end exclusive) and return
    per‑square accuracy for each board and overall average.
    """
    board_acc = []
    with torch.no_grad():
        for i in range(start, end):
            img, y = dataset[i]
            img = img.to(device)
            y   = y.to(device)  # ✅ also move labels if using on device

            logits = model(img.unsqueeze(0))
            pred = logits.argmax(-1).squeeze()
            true = model._labels_to_indices(y.unsqueeze(0)).squeeze()
            acc = (pred == true).float().mean().item()
            board_acc.append(acc)
    return board_acc

# ------------------------------------------------------------------ #
#  2. Visualisation for one board
# ------------------------------------------------------------------ #
def show_board(model, dataset, index: int, device="cpu"):
    raw_np = dataset.get_raw_image(index)
    img, y = dataset[index]
    img = img.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits = model(img.unsqueeze(0))

    pred = logits.argmax(-1).squeeze().view(8, 8).cpu().numpy()
    true = model._labels_to_indices(y.unsqueeze(0)).squeeze().view(8, 8).cpu().numpy()
    acc  = (pred == true).mean()

    # down‑scale huge photos for display
    if max(raw_np.shape[:2]) > 1024:
        import cv2
        scale = 1024 / max(raw_np.shape[:2])
        raw_np = cv2.resize(raw_np, dsize=None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].imshow(raw_np); ax[0].set_title("Original Image"); ax[0].axis("off")
    ax[1].imshow(true, cmap="viridis", vmin=0, vmax=12)
    ax[1].set_title("Ground Truth"); ax[1].axis("off")
    im = ax[2].imshow(pred, cmap="viridis", vmin=0, vmax=12)
    ax[2].set_title("Model Prediction"); ax[2].axis("off")
    # fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6, label="Piece ID")
    plt.suptitle(f"Val idx {index}  •  per‑square acc {acc:.3f}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  3. Main
# ------------------------------------------------------------------ #
def main(args):
    # reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # ---- load model --------------------------------------------------
    ckpt_path = Path("lightning_logs/version_6/checkpoints/"
                     "best-epoch-epoch=11-val_loss=val_loss=0.2299.ckpt")
    model = ChessCNNTransformer.load_from_checkpoint(str(ckpt_path), map_location="cpu").eval()

    # ---- load validation dataset ------------------------------------
    dm = ChessDataModule(
        dataroot="./extracted_data",
        batch_size=1,
        workers=0,
        px_resize=512
    )
    dm.setup("fit")
    val_set = dm.chess_val
    n_val   = len(val_set)

    # ---- range selection --------------------------------------------
    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else n_val
    start = max(0, start); end = min(end, n_val)
    if start >= end:
        raise ValueError("Invalid range: start must be < end.")

    # ---- run predictions on range -----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running predictions on {device}")
    model = model.to(device)
    accs = predict_range(model, val_set, start, end, device=device)
    print(f"Evaluated boards {start} … {end-1}  |  "
          f"mean per‑square accuracy = {np.mean(accs):.4f}")

    # ---- choose one board to visualise ------------------------------
    if args.vis_index is not None:
        vis_idx = args.vis_index % n_val
    else:
        vis_idx = random.randint(start, end - 1)

    show_board(model, val_set, vis_idx, device=device)


# ------------------------------------------------------------------ #
#  4. CLI
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser(description="Chess board prediction demo")
    parser.add_argument("--start", type=int, help="Start index (inclusive)")
    parser.add_argument("--end",   type=int, help="End index (exclusive)")
    parser.add_argument("--vis-index", type=int, help="Index to visualise (default: random in range)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
