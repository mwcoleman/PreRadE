import os, json, sys
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

def load_paths_dict(cfg='data_paths.json'):
    with open(cfg, 'r') as file:
        pd = json.loads(file.read()) # use `json.loads` to do the reverse
    return pd




if __name__=='__main__':

    args = parse_args(stage='ft')

    # Reproducibility
    pl.seed_everything(808, workers=True)
    
    ####
    ### DEBUG args

    if len(sys.argv)<2:
        args.train = 'mimic'
        args.test = 'mimic'
        # args.use_val_split = False
        # args.no_evaluation = False
        args.run_name='baseline-100k-delme2'
        args.project='mmRad-mimic'
        args.epochs = 0
        args.topk = 512 #10240
        # args.load_model = '/media/matt/data21/mmRad/checkpoints/FT/FT-baseline/encoder'#"/media/matt/data21/mmRad/checkpoints/PT/mlm-mfr-itm/encoder" #/media/matt/data21/mmRad/checkpoints/PT/12L-SWA-mlm_mfr_itm/backbone/epoch=54-step=91519.ckpt"  #"uclanlp/visualbert-vqa-coco-pre" # 
        args.log_offline = True

    args.run_name = args.load_model if args.run_name=='tasks' else args.run_name


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
    Train Dataset: {args.train}
    Train size: {'full' if args.topk==0 else args.topk}
    Test Dataset: {args.test}
    Image only?: {args.img_only}
        
    Learning Rate: {args.lr}
    Using Scheduler: {args.lr_scheduler}\n\n\n""")
    
    # Needed if using TokenizerFast:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


    stage = 'test' if args.no_finetune else 'fit'
    
    path_dict = load_paths_dict()

    args.load_model = os.path.join(path_dict['pt_checkpoint_root'], args.load_model, "encoder")

    dm = MMRadDM(args, path_dict)
    dm.setup(stage=stage)
    
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

    # Save CP & encoder after fine tunining (if any)
    if args.epochs > 0:
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

    # Eval
    if not args.no_evaluation:
        trainer.test(model, dataloaders=dm)
        wandb_logger.experiment.config['test_size'] = dm.test_size
    

    