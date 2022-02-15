import argparse
import pytorch_lightning as pl



def parse_args(stage):
    parser = argparse.ArgumentParser()

    ##### TRAINING #####
    parser.add_argument('--name', dest='run_name', default='tasks')
    parser.add_argument('--project', default='mmRad-mimic')
    parser.add_argument('--log_offline', default=False, type=bool)
    parser.add_argument('--seed', type=int, default=808, help='random seed')
    parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=125)
    parser.add_argument('--epochs', dest='epochs', type=int, default=200)
    parser.add_argument('--steps', dest='steps',default=200000, type=int)
    # base dir for pl framework checkpoint files and hf encoder files
    parser.add_argument('--save_cp_path', dest='save_cp_path', type=str, 
                        default='/media/matt/data21/mmRad/checkpoints/')
    parser.add_argument('--save_encoder', dest='save_encoder', default=True)
    # location to the pl framework checkpoint (e.g. further pretraining)
    parser.add_argument('--load_cp_path', dest='load_cp_path', default=None)
    ## Tasks ##
    parser.add_argument('--tasks', default="mlm,itm", type=str)



    ##### MODEL #####
    parser.add_argument('--load_model', dest='load_model', default="uclanlp/visualbert-vqa-coco-pre")
    parser.add_argument('--load_model_config', dest='load_model_config', default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float)
    parser.add_argument('--warmup_ratio', dest='warmup_ratio', default=0.15, type=float,
                        help='decimal fraction of total training steps to schedule warmup')
    parser.add_argument('--lr_scheduler', default=True, help='Whether to use lr scheduler')
    parser.add_argument('--tokenizer', default='bert-base-uncased')
    
    # Tx architecture
    parser.add_argument('--freeze', default=False, help='Freeze Tx encoder')
    parser.add_argument('--num_tx_layers', dest='num_tx_layers', default=12, type=int)
    parser.add_argument('--num_attention_heads', dest='num_attention_heads', default=12, type=int)
    parser.add_argument('--encoder_hidden_size', dest='encoder_hidden_size', default=768, type=int)
    parser.add_argument('--visual_embedding_dim', dest='visual_embedding_dim', default=2048, type=int)
    parser.add_argument('--extracted_ft_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    
    

    ## Classification only
    parser.add_argument('--img_only', dest='img_only', default=False, type=bool)
    parser.add_argument('--txt_only', dest='txt_only', default=False, type=bool)
    parser.add_argument('--easy_classification', default=False)
    ##### DATA #####

    # Data splits
    parser.add_argument("--dataset", dest='dataset', default='mimic')
    parser.add_argument("--txt_path", default='studies_with_splits.csv')
    parser.add_argument("--img_path", default='mimic_train_100k.tsv')
    parser.add_argument("--valid_data", dest='valid_data', default=None) # name of .tsv file
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--drop_last", dest='drop_last', default=True)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--topk", default=0, type=int)
    parser.add_argument("--val_topk", dest='val_topk', default=None)
    # Sizing
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', dest='valid_batch_size', type=int, default=64)
    # Data path
    parser.add_argument('--data_path', dest='data_path', default="/media/matt/data21/mmRad/MIMIC")
    
    ##### PL #####
    parser = pl.Trainer.add_argparse_args(parser)
    

    args = parser.parse_args()

    return args
