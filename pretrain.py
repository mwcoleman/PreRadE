from modules import MMRadForPretraining, MMRadDM
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import wandb

##
# TODO: This can be removed once pytorch-lightning issue #10408 is merged
# https://github.com/PyTorchLightning/pytorch-lightning/pull/10408
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings(
    "ignore", ".*DataModule.setup has already been called.*"
)
##


# Sample run 
if __name__=='__main__':
    from parameters import parse_args
    args = parse_args(stage='pt')
    ####

    ### DEBUG args
    args.train_split = 'mscoco_train'
    args.valid_split = 'mscoco_val'
    args.run_name='delme-mfr-mlm'
    args.epochs = 100
    args.topk = 5120
    args.load_model = None
    args.num_attention_heads = 12
    args.num_tx_layers = 6
    args.tasks = "['mlm', 'mfr']"
    # args.val_topk = None

    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad-coco-sweeps')
    wandb_logger.experiment.config.update(args)
    # # Used with sweep agent
    wandb.init(config=wandb_logger.experiment.config)


    dm = MMRadDM(args)
    dm.setup(stage='fit')

    
    if args.load_cp_path is None:
        model = MMRadForPretraining(args=args, train_size=dm.train_size)
    else:
        # Load a pretrained model for (further) pretraining
        print(f'Loading saved model from {args.load_cp_path}')
        model = MMRadForPretraining(args=args, train_size=dm.train_size).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size)
    
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_avg_loss",
        dirpath=args.save_cp_path + args.run_name + '/pl_framework/',
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_avg_loss')
    
    callbacks=[checkpoint_callback, lr_monitor, early_stopping]

    # Reproducibility
    pl.seed_everything(808, workers=True)

    # Training
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=callbacks, 
                         log_every_n_steps=10, max_epochs=args.epochs, deterministic=True, 
                         logger=wandb_logger, track_grad_norm=-1, fast_dev_run=False)

    print(f"\nBeginning training run with {args.topk} train, {args.val_topk} val examples from {args.train_split}. Training for {args.epochs} epochs...\n")
    
    trainer.fit(model, dm)
    
    # Save model states
    print(f"PL Model and state saved to {checkpoint_callback.best_model_path}")
    wandb_logger.experiment.config['pl_framework_path'] = checkpoint_callback.best_model_path

    if args.save_backbone:
        model.model.save_pretrained(save_directory=args.save_cp_path + args.run_name + '/backbone/')
        print(f"Tx backbone saved to {args.save_cp_path + args.run_name + '/backbone/'}")
        wandb_logger.experiment.config['backbone_path'] = args.save_cp_path + args.run_name + '/backbone/'