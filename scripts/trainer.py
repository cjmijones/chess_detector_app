import torch

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    from pytorch_lightning import Trainer
    from chess_dataloader import ChessDataModule
    from cnn_transformer_model import ChessCNNTransformer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    model = ChessCNNTransformer(
        lr=1e-4,
        freeze_backbone_epochs=3,
        freeze_transformer_epochs=2,
        unfreeze_cnn_blocks=3,
        unfreeze_transformer_layers=1,
        verbose=True
    )

    full_dm = ChessDataModule(
        dataroot="./extracted_data",
        batch_size=12,
        workers=4,
        px_resize=512
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        filename="best-epoch-{epoch:02d}-val_loss={val_loss:.4f}"
    )

    logger = TensorBoardLogger(
        save_dir="lightning_logs",   # where logs are saved
        name="chess_model"           # subfolder name
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",              # AMP = faster & less VRAM
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=full_dm)
