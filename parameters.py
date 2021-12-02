import argparse
import pytorch_lightning as pl

def parse_args(stage):
    parser = argparse.ArgumentParser()

    ##### TRAINING #####
    parser.add_argument('--name', dest='run_name', default='debug_4')
    parser.add_argument('--seed', type=int, default=808, help='random seed')
    parser.add_argument('--maxSeqLen', dest='max_seq_len', type=int, default=20)
    parser.add_argument('--epochs', dest='epochs', type=int, default=200)
    # base dir for pl framework checkpoint files and hf backbone files
    parser.add_argument('--savePath', dest='save_cp_path', type=str, 
                        default='/media/matt/data21/mmRad/checkpoints/PT/')
    parser.add_argument('--saveBackbone', dest='save_backbone', default=True)
    # location to the pl framework checkpoint (e.g. further pretraining)
    parser.add_argument('--loadPath', dest='load_cp_path', default=None)

    ##### MODEL #####
    parser.add_argument('--loadModelConfig', dest='load_model_config', default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loadModel', dest='load_model', default=None)
    parser.add_argument('--warmupRatio', dest='warmup_ratio', default=0.15, 
                        help='decimal fraction of total training steps to schedule warmup')
    parser.add_argument('--freeze', default=False)

    ##### DATA #####
    # Data splits
    parser.add_argument("--train", dest='train_split', default='mscoco_train')
    parser.add_argument("--valid", dest='valid_split', default='mscoco_val')
    parser.add_argument("--test", default=None)
    parser.add_argument("--dropLast", dest='drop_last', default=True)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--topk", default=5120)
    parser.add_argument("--topkVal", dest='val_topk', default=None)
    # Sizing
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--batchSizeVal', dest='valid_batch_size', type=int, default=256)
    # Data path
    parser.add_argument('--dataPath', dest='data_path', default="/media/matt/data21/mmRad/")
    
    ##### PL #####
    parser = pl.Trainer.add_argparse_args(parser)


    args = parser.parse_args()

    return args
