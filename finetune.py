import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging


from src.model import MMRadForClassification
from src.data import MMRadDM
from src.parameters import parse_args
from src.utils import MetricsCallback

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

    args = parse_args(stage='ft')

    # Reproducibility
    pl.seed_everything(808, workers=True)
    
    ####
    ### DEBUG args
    import sys
    if len(sys.argv)<2:
        args.dataset = 'openI'
        args.run_name='delme'
        args.epochs = 1
        args.topk = 512 #10240
        args.load_model = "/media/matt/data21/mmRad/checkpoints/PT/mlm-mfr-itm/encoder" #/media/matt/data21/mmRad/checkpoints/PT/12L-SWA-mlm_mfr_itm/backbone/epoch=54-step=91519.ckpt"  #"uclanlp/visualbert-vqa-coco-pre" # 
        args.freeze=False # Freeze the encoder (init from scratch)
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
        args.log_offline = True
        args.test_data = 'openI_all.tsv'
 
    print(f"""\n\n\nFinetuning with parameters: \n
    Run name: {args.run_name}
    Checkpoint loaded from: {args.load_cp_path} 
    Encoder loaded from: {args.load_model}
    Tokenizer: {args.tokenizer} 
    # Att Heads: {args.num_attention_heads}
    # Layers: {args.num_tx_layers} 
    Training for max steps / epochs: {args.steps} / {args.epochs}
    Batch size: {args.batch_size} 
    Max sequence length: {args.max_seq_len} 
    Train Dataset: {args.img_path}
    Train size: {'full' if args.topk==0 else args.topk}
    Test Dataset: {args.dataset}
        
    Learning Rate: {args.lr}
    Using Scheduler: {args.lr_scheduler}\n\n\n""")



    dm = MMRadDM(args)
    dm.setup(stage='fit')
    
    cp_path = os.path.join(args.save_cp_path,'FT',args.run_name,'pl_framework')
    encoder_path = os.path.join(args.save_cp_path,'FT',args.run_name,'encoder')


    if args.load_cp_path is None:
        model = MMRadForClassification(
                args=args, 
                train_size=dm.train_size, 
                n_classes=dm.num_classes, 
                labelset=dm.labelset)
    else:
        # Load a classification checkpoint
        print(f'Loading saved checkpoint from {args.load_cp_path}')
        model = MMRadForClassification(
                args=args, 
                train_size=dm.train_size, 
                n_classes=dm.num_classes, 
                labelset=dm.labelset
                ).load_from_checkpoint(
                    args.load_cp_path, 
                    args=args, 
                    train_size=dm.train_size, 
                    n_classes=dm.num_classes, 
                    labelset=dm.labelset
                    )
    
    ## Logging & Callbacks
    wandb_logger = WandbLogger(
        name=args.run_name, 
        project=args.project, 
        offline= args.log_offline
        )

    wandb_logger.watch(model)
    wandb_logger.experiment.config.update(args)
    

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cp_path,
        every_n_epochs=5,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    auroc_metrics = MetricsCallback(dm.num_classes)
    # swa = StochasticWeightAveraging()

    callbacks = [#checkpoint_callback, 
                 lr_monitor, 
                 auroc_metrics]

    trainer = pl.Trainer.from_argparse_args(
        args, 
        gpus=1, 
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10, 
        max_epochs=args.epochs, 
        deterministic=True,  
        track_grad_norm=-1,
        fast_dev_run=False, 
        benchmark=True,
        )
    


    trainer.fit(model, dm)

    # Eval
    if args.test_data is not None:
        trainer.test(model, test_dataloaders=dm)
        wandb_logger.experiment.config['test_size'] = dm.test_size
    
    # Save CP & encoder
    trainer.save_checkpoint(cp_path)
    print(f"Checkpoint saved to {cp_path}")
    wandb_logger.experiment.config['cp_path'] = cp_path
    if args.save_encoder:
        model.model.save_pretrained(save_directory=encoder_path)
        print(f"Encoder weights saved to {encoder_path}")
        wandb_logger.experiment.config['encoder_path'] = encoder_path

    # Log the dataset sizes
    wandb_logger.experiment.config['train_size'] = dm.train_size
    wandb_logger.experiment.config['valid_size'] = dm.valid_size
    