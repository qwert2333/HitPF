
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from src.layers.utils_training import FreezeClustering

def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
                dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
                filename="_{epoch}_{step}",
                # every_n_epochs=val_every_n_epochs,
                every_n_train_steps=500,
                save_top_k=-1,  # <--- this is important!
                save_weights_only=True,
            )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
        lr_monitor,
    ]
    if args.freeze_clustering:
            callbacks.append(FreezeClustering())
    return callbacks

def get_callbacks_eval(args):
    callbacks=[TQDMProgressBar(refresh_rate=1)]
    return callbacks