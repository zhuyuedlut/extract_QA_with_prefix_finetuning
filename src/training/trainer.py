import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers


def generate_trainer(args: argparse.Namespace):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=6,
    )
    callbacks = [checkpoint_callback, early_stop_callback]

    logger = pl_loggers.TensorBoardLogger("logs/")

    trainer = Trainer(
        args,
        gpus=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=20,
        logger=logger,
    )

    return trainer
