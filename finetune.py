from modules import MMRadForClassification, MMRadDM, MMRadForPretraining
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, QuantizationAwareTraining
import os
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

    # Reproducibility
    pl.seed_everything(808, workers=True)
    
    ####
    ### DEBUG args
    import sys
    if len(sys.argv)<2:
        args.dataset = 'mimic'
        args.run_name='mlm_mfr_itm-30-Val_6k'
        args.epochs = 30
        args.topk = 0 #10240
        args.load_model = "/media/matt/data21/mmRad/checkpoints/PT/12L-SWA-mlm_mfr_itm/backbone/epoch=54-step=91519.ckpt"  #"uclanlp/visualbert-vqa-coco-pre" # 
        args.freeze=False # Freeze the backbone (init from scratch)
        args.img_only = False
        args.img_path = 'mimic_val_100k.tsv'
        args.num_tx_layers = 12
        args.num_att_heads = 12
        args.lr = 5e-5
        # args.max_hrs = 4
        # args.lr_scheduler = False
        args.val_topk = 0
        args.batch_size=64
        args.max_seq_len=125
        # args.img_only = True
        # args.txt_only = True
        # args.easy_classification = True
        # args.log_offline = True
    
    dm = MMRadDM(args)
    dm.setup(stage='fit')
    
    cp_path = os.path.join(args.save_cp_path,'FT',args.run_name,'pl_framework')
    backbone_path = os.path.join(args.save_cp_path,'FT',args.run_name,'backbone')


    if args.load_cp_path is None:
        model = MMRadForClassification(args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset)
    else:
        # Load a classification checkpoint
        print(f'Loading saved model from {args.load_cp_path}')
        model = MMRadForClassification(args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size, n_classes=dm.num_classes, labelset=dm.labelset)
    
    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad-mimic', offline= args.log_offline)
    wandb_logger.watch(model)
    wandb_logger.experiment.config.update(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cp_path,
        every_n_epochs=5,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    from modules import LogValMetrics#, TimeIt
    auroc_metrics = LogValMetrics(dm.num_classes)
    from pytorch_lightning.callbacks import StochasticWeightAveraging
    swa = StochasticWeightAveraging()
    # Reproducibility
    pl.seed_everything(808, workers=True)

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=[checkpoint_callback, lr_monitor, auroc_metrics, swa], 
                         log_every_n_steps=10, max_epochs=args.epochs, deterministic=True, 
                         logger=wandb_logger, track_grad_norm=-1, fast_dev_run=False, benchmark=True,
                         max_time={"hours": args.max_hrs})
    
    print(f"\nBeginning training run with {args.topk} training examples from {args.dataset}. Training for {args.epochs} epochs...\n")
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
        model.model.save_pretrained(save_directory=backbone_path)
        print(f"Tx backbone saved to {backbone_path}")
        wandb_logger.experiment.config['backbone_path'] = backbone_path
