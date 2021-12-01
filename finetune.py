from modules import *

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

if __name__=='__main__':
    #### Args
    from argparse import ArgumentParser
    parser = ArgumentParser()

    ## Program level
    parser.add_argument('--name', dest='run_name', default='FT-mlm')
    parser.add_argument('--seed', type=int, default=808, help='random seed')
    parser.add_argument('--maxSeqLen', dest='max_seq_len', type=int, default=20)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    # base dir for pl framework checkpoint files
    parser.add_argument('--savePath', dest='save_cp_path', type=str, 
                        default='/media/matt/data21/mmRad/checkpoints/FT/')
    parser.add_argument('--loadPath', dest='load_cp_path', default=None)

    ## Model specific
    parser = MMRadForClassification.add_model_specific_args(parser)
    ## Data specific
    parser = MMRadDM.add_model_specific_args(parser)
    ## Trainer specific
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    ####
    ### DEBUG args
    args.train_split = 'cub_train'
    args.valid_split = 'cub_valid'
    args.run_name='class_cub_run'
    args.topk = 500
    args.val_topk = 500

    dm = MMRadDM(args)
    dm.setup(stage='fit')

    
    if args.load_cp_path is None:
        model = MMRadForClassification(args=args, train_size=dm.train_size, num_classes=dm.num_classes)
    else:
        # Load a pretrained model for (further) pretraining
        print(f'Loading saved model from {args.load_cp_path}')
        model = MMRadForPretraining(args=args, train_size=dm.train_size).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size)
    
    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad')
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_cp_path + args.run_name + '/pl_framework/',
        every_n_epochs=4
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
   
   
    # Reproducibility
    pl.seed_everything(808, workers=True)

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=[checkpoint_callback, lr_monitor], 
                         log_every_n_steps=10, max_epochs=args.epochs, deterministic=False, 
                         logger=wandb_logger, 
                         track_grad_norm=-1, fast_dev_run=False)
    
    print(f"\nBeginning training run with {args.topk} training examples from {args.train_split}. Training for {args.epochs} epochs...\n")
    trainer.fit(model, dm)

 # Save model states
    print(f"PL Model and state saved to {checkpoint_callback.best_model_path}")
    
    if args.save_backbone:
        model.model.save_pretrained(save_directory=args.save_cp_path + args.run_name + '/backbone/')
        print(f"Tx backbone saved to {args.save_cp_path + args.run_name + '/backbone/'}")