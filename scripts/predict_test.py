# scripts/evaluate_test_set.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

from cnn_transformer_model import ChessCNNTransformer
from chess_dataloader import ChessDataModule

def evaluate_test_set():
    # -- Setup -- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Load model -- #
    ckpt_path = Path("lightning_logs/version_6/checkpoints/"
                     "best-epoch-epoch=11-val_loss=val_loss=0.2299.ckpt")
    model = ChessCNNTransformer.load_from_checkpoint(str(ckpt_path)).to(device).eval()

    # -- Load test dataset -- #
    dm = ChessDataModule(
        dataroot="./extracted_data",
        batch_size=1,
        workers=0,
        px_resize=512
    )
    dm.setup("test")
    test_set = dm.chess_test

    # -- Run inference -- #
    all_preds, all_trues = [], []
    incorrect_counts = []

    for i in tqdm(range(len(test_set)), desc="Predicting"):
        img, y = test_set[i]
        img, y = img.to(device), y.to(device)

        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            pred = logits.argmax(-1).squeeze()             # (64,)
            true = model._labels_to_indices(y.unsqueeze(0)).squeeze()

        all_preds.extend(pred.view(-1).cpu().numpy())
        all_trues.extend(true.view(-1).cpu().numpy())

        incorrect = (pred != true).sum().item()
        incorrect_counts.append(incorrect)

    # -- Accuracy -- #
    square_acc = np.mean(np.array(all_preds) == np.array(all_trues))
    print(f"\n✅ Per-square accuracy on test set: {square_acc:.4f}")

    # -- Confusion Matrix -- #
    cm = confusion_matrix(all_trues, all_preds, labels=np.arange(13))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.arange(13), yticklabels=np.arange(13))
    plt.title("Confusion Matrix (Piece Class)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # -- Error Histogram -- #
    max_incorrect = max(incorrect_counts)
    bins = np.arange(0, max_incorrect + 2) - 0.5
    plt.figure(figsize=(8, 5))
    plt.hist(incorrect_counts, bins=bins, edgecolor="black")
    plt.xticks(np.arange(max_incorrect + 1))
    plt.xlabel("Incorrect Squares per Board")
    plt.ylabel("Number of Boards")
    plt.title("Board Error Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_test_set()
