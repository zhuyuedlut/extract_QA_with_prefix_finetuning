import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def generate_trainer(args: argparse.Namespace):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='f1',
        save_top_k=1,
        mode='max',
    )
    early_stop_callback = EarlyStopping(
        monitor='f1',
        mode='max',
        patience=6,
    )
    callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(callbacks=callbacks, log_every_n_steps=20)

    return trainer
