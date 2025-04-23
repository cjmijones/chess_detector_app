# chess_cnn_transformer.py
import os, csv, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import convnext_base
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import Accuracy


class ChessCNNTransformer(pl.LightningModule):
    """
    ConvNeXt‑B backbone + Transformer encoder that supports gradual unfreezing.
    The implementation is streamlined for speed:
      • only board‑level accuracy during training
      • board‑ and square‑level accuracy during validation
      • metrics computed with torchmetrics on‑GPU
      • plots generated every 5 epochs
    """

    # ------------------------------------------------------------------ #
    #  INIT
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        lr: float = 2e-4,
        num_layers: int = 4,
        num_heads: int = 8,
        embed_dim: int = 1024,
        freeze_backbone_epochs: int = 3,
        freeze_transformer_epochs: int = 3,
        unfreeze_cnn_blocks: int = 1,
        unfreeze_transformer_layers: int = 1,
        verbose: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── backbone ────────────────────────────────────────────────────
        cnn = convnext_base(weights="IMAGENET1K_V1")
        self.cnn_backbone = nn.Sequential(*list(cnn.children())[:-2])
        self.register_buffer("_total_cnn_blocks",
                             torch.tensor(len(list(self.cnn_backbone.children()))))

        # ── transformer encoder ────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.register_buffer("_total_tx_layers", torch.tensor(num_layers))

        # ── head & tokens ───────────────────────────────────────────────
        self.square_tokens = nn.Parameter(torch.randn(64, embed_dim))
        self.head = nn.Linear(embed_dim, 13)  # 13 piece classes incl. “empty”

        # ── metrics ─────────────────────────────────────────────────────
        self.train_board_acc = Accuracy(task="binary")  # treats board‑correct as 0/1
        self.val_board_acc   = Accuracy(task="binary")
        self.val_square_acc  = Accuracy(task="multiclass", num_classes=13, average="micro")

        # ── bookkeeping ────────────────────────────────────────────────
        self.verbose = verbose
        os.makedirs("training_plots", exist_ok=True)
        self.metrics_csv = os.path.join("training_plots", "training_metrics.csv")
        with open(self.metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss",
                 "train_board_acc", "val_board_acc", "val_square_acc"]
            )
        self._last_train_loss = None

        # ── freeze everything (head stays trainable) ───────────────────
        for p in self.parameters():
            p.requires_grad = False
        for p in list(self.head.parameters()) + [self.square_tokens]:
            p.requires_grad = True

    # ------------------------------------------------------------------ #
    #  FORWARD
    # ------------------------------------------------------------------ #
    def forward(self, x):
        B = x.size(0)
        feat_map = self.cnn_backbone(x)                # (B,C,H,W)
        tokens   = feat_map.flatten(2).permute(0, 2, 1)
        queries  = self.square_tokens.unsqueeze(0).expand(B, -1, -1)
        encoded  = self.transformer(torch.cat([queries, tokens], 1))[:, :64]
        logits   = self.head(encoded)                  # (B, 64, 13)
        return logits

    # ------------------------------------------------------------------ #
    #  LOSS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _labels_to_indices(labels):
        if labels.ndim == 2:                # (B,64) or (B,832)
            if labels.shape[1] == 64:
                return labels.long()
            if labels.shape[1] == 64 * 13:
                return labels.view(labels.size(0), 64, 13).argmax(-1)
        if labels.ndim == 3:                # (B,64,13)
            return labels.argmax(-1)
        raise ValueError(f"Unsupported label shape {labels.shape}")

    def loss_fn(self, logits, labels):
        target = self._labels_to_indices(labels).view(-1)
        return F.cross_entropy(logits.view(-1, 13), target)

    # ------------------------------------------------------------------ #
    #  TRAIN / VAL STEPS
    # ------------------------------------------------------------------ #
    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y)

        target = self._labels_to_indices(y)
        preds  = logits.argmax(-1)
        board_correct = (preds == target).all(-1).float()       # (B,)

        self.train_board_acc.update(board_correct, torch.ones_like(board_correct))
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        self._last_train_loss = loss.detach().cpu().item()
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y)

        target = self._labels_to_indices(y)
        preds  = logits.argmax(-1)

        board_correct = (preds == target).all(-1).float()
        val_acc_batch = (preds.view(-1) == target.view(-1)).float().mean()
        self.val_board_acc.update(board_correct, torch.ones_like(board_correct))
        self.val_square_acc.update(preds.view(-1), target.view(-1))
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        self.log("val_square_acc_step", val_acc_batch, prog_bar=True)

    # ------------------------------------------------------------------ #
    #  OPTIMISER
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=self.trainer.max_epochs,
            pct_start=0.2, div_factor=25.0, final_div_factor=10.0
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    # ------------------------------------------------------------------ #
    #  UNFREEZE HOOK
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.freeze_backbone_epochs:
            k = self.hparams.unfreeze_cnn_blocks
            for blk in list(self.cnn_backbone.children())[-k:]:
                for p in blk.parameters(): p.requires_grad = True
            if self.verbose:
                print(f"[INFO] Unfroze last {k} ConvNeXt blocks")

        if self.current_epoch == self.hparams.freeze_transformer_epochs:
            l = self.hparams.unfreeze_transformer_layers
            for layer in self.transformer.layers[-l:]:
                for p in layer.parameters(): p.requires_grad = True
            if self.verbose:
                print(f"[INFO] Unfroze last {l} Transformer layers")

    # ------------------------------------------------------------------ #
    #  EPOCH‑END LOGGING
    # ------------------------------------------------------------------ #
    def on_train_epoch_end(self):
        # store for CSV; reset metric
        self._epoch_train_board = self.train_board_acc.compute().item()
        self.train_board_acc.reset()

    def on_validation_epoch_end(self):
        # metrics
        v_loss = self.trainer.callback_metrics["val_loss"].item()
        v_board = self.val_board_acc.compute().item()
        v_square = self.val_square_acc.compute().item()
        self.val_board_acc.reset()
        self.val_square_acc.reset()

        train_loss = self._last_train_loss if hasattr(self, "_last_train_loss") else float("nan")
        train_board = getattr(self, "_epoch_train_board", float("nan"))  # ✅ safe fallback

        # write CSV
        with open(self.metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.current_epoch,
                train_loss,
                v_loss,
                train_board,
                v_board,
                v_square
            ])

        # plot every 5 epochs
        if (self.current_epoch + 1) % 1 == 0:
            self.plot_training()

    # ------------------------------------------------------------------ #
    #  PLOT
    # ------------------------------------------------------------------ #
    def plot_training(self):
        if not os.path.exists(self.metrics_csv):
            return

        ep, tl, vl, tba, vba, vsa = [], [], [], [], [], []
        with open(self.metrics_csv, newline="") as f:
            reader = csv.reader(f); next(reader)
            for row in reader:
                if len(row) < 6:            # malformed — skip
                    continue
                # convert safely
                try:
                    ep.append(int(row[0]))
                    tl.append(float(row[1]))        # "nan" → float("nan") works
                    vl.append(float(row[2]))
                    tba.append(float(row[3]))
                    vba.append(float(row[4]))
                    vsa.append(float(row[5]))
                except ValueError:
                    continue                        # skip bad line

        # Need at least *one* complete epoch (train+val) to plot
        if not (ep and tl and vl):
            return
        if not (len(ep) == len(tl) == len(vl) == len(tba) == len(vba) == len(vsa)):
            return

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(ep, tl, label="train")
        ax[0].plot(ep, vl, label="val")
        ax[0].set_title("Loss"); ax[0].legend()

        ax[1].plot(ep, tba, label="board train")
        ax[1].plot(ep, vba, label="board val")
        ax[1].plot(ep, vsa, label="square val", ls="--")
        ax[1].set_title("Accuracy"); ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join("training_plots", f"epoch_{ep[-1]:03d}.png"))
        plt.close(fig)

