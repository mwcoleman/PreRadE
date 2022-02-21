import os
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.model import MMRadForPretraining
from src.data import MMRadDM
from src.parameters import parse_args
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
    
    import sys
    args = parse_args(stage='pt')
    pl.seed_everything(808, workers=True)
    ### DEBUG args
    if len(sys.argv)<2:
        # args.max_steps = 2000
        # args.topk = 5120
        args.load_model = "uclanlp/visualbert-vqa-coco-pre"
        args.tasks = "oovm,mfr,itm"
        # args.log_offline = True

    # Define run name:
    args.run_name = args.tasks.replace(',','-') if args.run_name == 'tasks' else args.run_name

    # Needed if using TokenizerFast:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    print(f"""\n\n\nPretraining with parameters: \n
    Run name: {args.run_name}
    Tasks: {args.tasks}
    Checkpoint loaded from: {args.load_cp_path} 
    Encoder loaded from: {args.load_model}
    Tokenizer: {args.tokenizer} 
    # Att Heads: {args.num_attention_heads}
    # Layers: {args.num_tx_layers} 
    Training for max steps / epochs: {args.steps} / {args.epochs}
    Batch size: {args.batch_size} 
    Max sequence length: {args.max_seq_len} 
    Dataset: {args.dataset}
    Subset?: {args.topk}
    
    Learning Rate: {args.lr}
    Using Scheduler: {args.lr_scheduler}\n\n\n""")

    # Logging & Callbacks
    wandb_logger = WandbLogger(
        name=args.run_name, 
        project='mmRad-Pretraining', 
        offline=args.log_offline
        )

    wandb_logger.experiment.config.update(args)
    # Used with sweep agent
    wandb.init(config=wandb_logger.experiment.config)

    dm = MMRadDM(args)
    dm.setup(stage='fit')
    
    print(f"\nTrain/Val splits: {dm.train_size} / {dm.valid_size}")
    
    if args.load_cp_path is None:
        model = MMRadForPretraining(
            args=args,
            train_size=dm.train_size, 
            tokenizer=args.tokenizer
            )
    else:
        # Load a pretrained model for (further) pretraining
        print(f'Loading saved model from {args.load_cp_path}')
        model = MMRadForPretraining(
            args=args, 
            train_size=dm.train_size,
            tokenizer=args.tokenizer
            ).load_from_checkpoint(
                args.load_cp_path, 
                args=args, 
                train_size=dm.train_size)
    
    wandb_logger.watch(model)

    cp_path = os.path.join(args.save_cp_path,'PT',args.run_name,'pl_framework')
    encoder_path = os.path.join(args.save_cp_path,'PT',args.run_name,'encoder')
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_avg_loss",
        dirpath=cp_path,
        every_n_epochs=5,
        save_top_k=-1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks=[checkpoint_callback, lr_monitor]

    # Training
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1, 
        callbacks=callbacks,
        logger=wandb_logger, 
        log_every_n_steps=10, 
        # max_epochs=args.epochs,
        max_steps=args.steps, 
        deterministic=True, 
        track_grad_norm=-1, 
        fast_dev_run=False, 
        benchmark=True
    )


    
    trainer.fit(model, dm)
      
    # Save model states
    print(f"PL Model and state (best val loss) saved to {checkpoint_callback.best_model_path}")
    wandb_logger.experiment.config['pl_framework_path'] = checkpoint_callback.best_model_path
    
    # Log the dataset sizes
    wandb_logger.experiment.config['train_size'] = dm.train_size
    wandb_logger.experiment.config['valid_size'] = dm.valid_size  


    if args.save_encoder:
        # Save the encoder at the last epoch
        model.model.save_pretrained(save_directory=encoder_path)
        print(f"Tx encoder (at last epoch) saved to {encoder_path}")
        wandb_logger.experiment.config['encoder_path'] = encoder_path

        # Also save encoder from best val
        # TODO: FIX- 'error; is a dir' error
        model = MMRadForPretraining(
            args=args, 
            train_size=dm.train_size,
            tokenizer=args.tokenizer
            ).load_from_checkpoint(
                checkpoint_callback.best_model_path,
                args=args, 
                train_size=dm.train_size
                )

        model.model.save_pretrained(save_directory=os.path.join(encoder_path,'best_val'))