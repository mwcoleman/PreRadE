from modules import CheckpointEveryNEpochs, MMRadForPretraining, MMRadDM
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

    ### DEBUG args
    if len(sys.argv)<2:
        args.train_split = 'mimic_train'
    # args.valid_split = 'mscoco_val' # not used
        args.run_name='delme'
        args.epochs = 120
        args.topk = 512
        args.load_model = None
        args.num_attention_heads = 12
        args.num_tx_layers = 12
        args.tasks = "['mlm','itm']"
        args.lr = 5e-5
        # args.log_offline = True
        args.max_seq_len = 50
        args.batch_size = 128
        args.valid_batch_size = 128
        # args.val_topk = None  # not used?
        # args.load_model= "uclanlp/visualbert-vqa-coco-pre"
        # args.tokenizer='emilyalsentzer/Bio_ClinicalBERT'
        # args.load_cp_path = '/media/matt/data21/mmRad/checkpoints/PT/PT-mlmitm-12hr-benchmark/pl_framework/epoch=23-step=9191.ckpt'
    # Logging & Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project='mmRad-mimic', offline=args.log_offline)
    wandb_logger.experiment.config.update(args)
    # # Used with sweep agent
    wandb.init(config=wandb_logger.experiment.config)

        # Debug grads

    dm = MMRadDM(args)
    dm.setup(stage='fit')


   
    
    model = MMRadForPretraining(args=args, train_size=dm.train_size,tokenizer=args.tokenizer).load_from_checkpoint('/media/matt/data21/mmRad/checkpoints/PT/PT-mlmitm-12hr-benchmark-continued/pl_framework/epoch=79-step=30639.ckpt', args=args, train_size=dm.train_size)
    model.model.save_pretrained(save_directory=args.save_cp_path + args.run_name + '/backbone-epoch79/')
    print(f"Tx backbone (at last epoch) saved to {args.save_cp_path + args.run_name + '/backbone-epoch79/'}")
