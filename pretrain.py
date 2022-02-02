from modules import  MMRadForPretraining, MMRadDM
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
    import sys
    args = parse_args(stage='pt')
    ####
    # Reproducibility
    pl.seed_everything(808, workers=True)
    ### DEBUG args
    if len(sys.argv)<2:
        args.dataset = 'mimic'
    # args.valid_split = 'mscoco_val' # not used
        args.run_name='12L-SWA-mlm_mfr_itm'
        args.epochs = 200
        args.topk = 0
        args.load_model = "uclanlp/visualbert-vqa-coco-pre"
        args.num_attention_heads = 12
        args.num_tx_layers = 12
        args.tasks = "['mlm','mfr','itm']"
        args.lr = 5e-5
        # args.log_offline = True
        args.max_seq_len = 125
        args.batch_size = 64
        args.valid_batch_size = 64
        args.max_hrs = 11
        # args.val_topk = None  # not used?
        # args.load_model= "uclanlp/visualbert-vqa-coco-pre"
        # args.tokenizer='emilyalsentzer/Bio_ClinicalBERT'
        # args.load_cp_path = '/media/matt/data21/mmRad/checkpoints/PT/PT-mlmitm-12hr-benchmark/pl_framework/epoch=23-step=9191.ckpt'
        
    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad-Pretraining', offline=args.log_offline)
    wandb_logger.experiment.config.update(args)
    # # Used with sweep agent
    wandb.init(config=wandb_logger.experiment.config)

        # Debug grads

    dm = MMRadDM(args)
    dm.setup(stage='fit')

    
    if args.load_cp_path is None:
        model = MMRadForPretraining(args=args, train_size=dm.train_size, tokenizer=args.tokenizer)
    else:
        # Load a pretrained model for (further) pretraining
        print(f'Loading saved model from {args.load_cp_path}')
        model = MMRadForPretraining(args=args, train_size=dm.train_size,tokenizer=args.tokenizer).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size)
    
    wandb_logger.watch(model)
    import os

    cp_path = os.path.join(args.save_cp_path,'PT',args.run_name,'pl_framework')
    backbone_path = os.path.join(args.save_cp_path,'PT',args.run_name,'backbone')
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_avg_loss",
        dirpath=cp_path,
        every_n_epochs=5,
        save_top_k=-1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    # early_stopping = EarlyStopping(monitor='val_avg_loss')
    from pytorch_lightning.callbacks import StochasticWeightAveraging
    # from modules import TimeIt
    swa = StochasticWeightAveraging()
    
    from pytorch_lightning.callbacks import QuantizationAwareTraining
    qat = QuantizationAwareTraining(input_compatible=False)
    callbacks=[checkpoint_callback, lr_monitor, swa]#, early_stopping]

    # Training
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=callbacks, 
                         log_every_n_steps=10, max_epochs=args.epochs, deterministic=True, 
                         logger=wandb_logger, track_grad_norm=-1, fast_dev_run=False, benchmark=True,
                         max_time={"hours": args.max_hrs})

    print(f"\nBeginning training run with {args.topk} train, {args.val_topk} val examples from {args.dataset}. Training for {args.epochs} epochs...\n")
    
    trainer.fit(model, dm)
    
    # Save model states
    print(f"PL Model and state (best val loss) saved to {checkpoint_callback.best_model_path}")
    wandb_logger.experiment.config['pl_framework_path'] = checkpoint_callback.best_model_path
    
    if args.save_backbone:
        model.model.save_pretrained(save_directory=args.save_cp_path + args.run_name + '/backbone/')
        print(f"Tx backbone (at last epoch) saved to {backbone_path}")
        wandb_logger.experiment.config['backbone_path'] = backbone_path

        # Also save backbone from best val
        model = MMRadForPretraining(args=args, 
                                    train_size=dm.train_size,
                                    tokenizer=args.tokenizer).load_from_checkpoint(checkpoint_callback.best_model_path,
                                                                                   args=args, train_size=dm.train_size)
        model.model.save_pretrained(save_directory=os.path.join(backbone_path,'best_val'))