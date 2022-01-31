from modules import MMRadForClassification, MMRadDM, MMRadForPretraining
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

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

    from parameters import parse_args
    args = parse_args(stage='ft')

    ####
    ### DEBUG args
    import sys
    if len(sys.argv)<2:
        args.train_split = 'mimic_train'
        # args.valid_split = 'cub_valid'
        args.run_name='FT-Mimic-MLM-ITM-30e-All-AP'
        args.epochs = 30
        args.topk = 0 #10240
        args.load_model = "/media/matt/data21/mmRad/checkpoints/PT/PT-mlmitm-12hr-benchmark-continued/backbone-epoch79/" # "uclanlp/visualbert-vqa-coco-pre" 
        args.freeze=False # Freeze the backbone (init from scratch)
        args.img_only = False
        args.lr = 5e-5
        # args.lr_scheduler = False
        args.val_topk = None
        args.batch_size=128
        # args.img_only = True
        # args.txt_only = True
        # args.easy_classification = True
        # args.log_offline = True
    
    dm = MMRadDM(args)
    dm.setup(stage='fit')
    
    if args.load_cp_path is None:
        model = MMRadForClassification(args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset)
    else:
        # Load a pretrained model for (further) training
        print(f'Loading saved model from {args.load_cp_path}')
        # TODO: Get rid of redundant arguments passed in..
        model = MMRadForClassification(args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset)
    
    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad-mimic', offline= args.log_offline)
    wandb_logger.watch(model)
    wandb_logger.experiment.config.update(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_cp_path + args.run_name + '/pl_framework/',
        every_n_epochs=10,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    from modules import LogValMetrics
    auroc_metrics = LogValMetrics()

    # Reproducibility
    pl.seed_everything(808, workers=True)

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=[checkpoint_callback, lr_monitor, auroc_metrics], 
                         log_every_n_steps=10, max_epochs=args.epochs, deterministic=True, 
                         logger=wandb_logger, track_grad_norm=-1, fast_dev_run=False)
    
    print(f"\nBeginning training run with {args.topk} training examples from {args.train_split}. Training for {args.epochs} epochs...\n")
    trainer.fit(model, dm)

    # Save model states

    print(f"PL Model and state saved to {checkpoint_callback.best_model_path}")
    wandb_logger.experiment.config['pl_framework_path'] = checkpoint_callback.best_model_path
    
    if args.save_backbone:
        model = MMRadForClassification(args=args,
                                       train_size=dm.train_size, 
                                       n_classes=dm.num_classes).load_from_checkpoint(checkpoint_callback.best_model_path,
                                                                                        args=args,
                                                                                        train_size=dm.train_size, 
                                                                                        n_classes=dm.num_classes)
        model.model.save_pretrained(save_directory=args.save_cp_path + args.run_name + '/backbone/')
        print(f"Tx backbone saved to {args.save_cp_path + args.run_name + '/backbone/'}")
        wandb_logger.experiment.config['backbone_path'] = args.save_cp_path + args.run_name + '/backbone/'
