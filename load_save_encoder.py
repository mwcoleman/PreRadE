import os, re
import pytorch_lightning as pl
from src.model import MMRadForClassification, MMRadDM, MMRadForPretraining
from src.parameters import parse_args

# Edit these as needed to load specific checkpoint
ROOT = '/media/matt/data21/mmRad/checkpoints/PT'
run_name = '12L-SWA-mlm_mfr_itm'
epoch_number = 54

if __name__=='__main__':
    import sys
    args = parse_args(stage='pt')
    #### Just keep this section the same as pretrain.py, 
    #### pl CP loading is glitchy otherwise
    pl.seed_everything(808, workers=True)
    args.dataset = 'mimic'
    args.run_name='12L-SWA-mlm_mfr_itm'
    args.epochs = 200
    args.topk = 512
    args.load_model = None #"uclanlp/visualbert-vqa-coco-pre"
    args.num_attention_heads = 12
    args.num_tx_layers = 12
    args.tasks = "['mlm','mfr','itm']"
    args.lr = 5e-5
    args.max_seq_len = 125
    args.batch_size = 64
    args.valid_batch_size = 64
    args.max_hrs = 11
    dm = MMRadDM(args)
    dm.setup(stage='fit')

    ## Define 
    numre = re.compile(r'epoch=(\d\d?)')


    saved_cp_paths = os.listdir(os.path.join(ROOT,run_name, 'pl_framework'))
    epoch_nums = [int(re.findall(numre,e)[0]) for e in saved_cp_paths]

    print(epoch_nums)

    # Load and save chosen epoch
    try:
        my_cp_path = saved_cp_paths[epoch_nums.index(epoch_number)]
    except:
        print("Error, epoch number not found in CP's")
        
    args.load_cp_path = os.path.join(ROOT,run_name,'pl_framework',my_cp_path)
    

    print(f'Loading saved model from {args.load_cp_path}')
    model = MMRadForPretraining(args=args, train_size=dm.train_size,tokenizer=args.tokenizer).load_from_checkpoint(args.load_cp_path, args=args, train_size=dm.train_size)
    print(f"saving encoder..")
    model.model.save_pretrained(save_directory=os.path.join(ROOT,run_name,'encoder',my_cp_path))
    print(f"encoder from cp {my_cp_path} saved to {os.path.join(ROOT,run_name,'encoder',my_cp_path)}")