import os, json, sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from transformers import EarlyStoppingCallback
import wandb

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
        args.train = 'mimic_5'
        args.test = 'mimic'
        # args.use_val_split = False
        # args.no_evaluation = False
        args.run_name='tasks'
        args.project='mmRad-mimic'
        args.epochs = 0
        args.topk = 5120 #10240
        # args.load_model = '/media/matt/data21/mmRad/checkpoints/FT/FT-baseline/encoder'#"/media/matt/data21/mmRad/checkpoints/PT/mlm-mfr-itm/encoder" #/media/matt/data21/mmRad/checkpoints/PT/12L-SWA-mlm_mfr_itm/backbone/epoch=54-step=91519.ckpt"  #"uclanlp/visualbert-vqa-coco-pre" # 
        args.log_offline = False
    
    # Needed if using TokenizerFast:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    stage = 'test' if args.no_finetune else 'fit'
    
    path_dict = load_paths_dict()



    ## Load encoder path(s)
    if args.load_model=="all":
        
        models = [(model_name,os.path.join(path_dict['pt_checkpoint_root'], model_name, "encoder"))
                   for model_name in os.listdir(path_dict['pt_checkpoint_root'])]
        print(f"Running on all models: {[m[0] for m in models]}")
        # encoder_paths = [os.path.join(path_dict['pt_checkpoint_root'], model_name, "encoder")
        #                  for model_name in os.listdir(path_dict['pt_checkpoint_root'])]
    elif args.load_model=="all_and_baseline":
        models = []
        models.append(('vbert',"uclanlp/visualbert-vqa-coco-pre"))
        models.append(('scratch',"scratch"))
        models = models + [(model_name,os.path.join(path_dict['pt_checkpoint_root'], model_name, "encoder"))
                   for model_name in os.listdir(path_dict['pt_checkpoint_root'])]


        print(f"Running on all models: {[m[0] for m in models]}")

    elif args.load_model in os.listdir(path_dict['pt_checkpoint_root']):
        models = [(args.load_model,os.path.join(path_dict['pt_checkpoint_root'], args.load_model, "encoder"))]
    else:
        models = [(args.load_model[:10], args.load_model)]
    ### Main Loop (Train, Test, Save a model):
    for i,(model_name,model_path) in enumerate(models):

        
        args.load_model = model_path

        appended = args.train if args.epochs > 0 else "_zero"

        # If run name specified and all models run; then tack it on to the end of each model name
        # Else it is run_name.
        log_run_name = model_name + args.run_name if ((args.run_name=='tasks')
                        or (len(models)>1)) else args.run_name

        ### Print run config
        print(f"\n-----Run {i} of {len(models)}-----")
        print(f"""\n\n\nFinetuning with parameters: \n
        Run name: {log_run_name}
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
        Text only?: {args.txt_only}

        Learning Rate: {args.lr}
        Using Scheduler: {args.lr_scheduler}\n\n\n""")
        ## Load data
        dm = MMRadDM(args, path_dict)
        dm.setup(stage=stage)

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
            name=log_run_name, 
            project=args.project, 
            offline= args.log_offline, reinit=True,
            save_dir='/media/matt/data21/mmRad/wandb/'
            )

        wandb_logger.watch(model)
        wandb_logger.experiment.config.update(args)
        

        # checkpoint_callback = ModelCheckpoint(
        #     monitor="val_loss",
        #     dirpath=cp_path,
        #     every_n_epochs=5,
        #     save_top_k=-1
        # )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        auroc_metrics = MetricsCallback(
            train_size=dm.train_size,
            valid_size=dm.valid_size,
            n_classes=dm.num_classes)
        from pytorch_lightning.callbacks import EarlyStopping
        callbacks = [#checkpoint_callback, 
                    lr_monitor, 
                    auroc_metrics]#, EarlyStopping(monitor="val_loss", mode="min", patience=5)]
        
        ## Train
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
            # Save-paths
            cp_path = os.path.join(args.save_cp_path,'FT',log_run_name,'pl_framework')
            encoder_path = os.path.join(args.save_cp_path,'FT',log_run_name,'encoder')

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

        # Eval- sorry for the double negative
        if not args.no_evaluation:
            trainer.test(model, dataloaders=dm)
            wandb_logger.experiment.config['test_size'] = dm.test_size
        wandb.finish()

    