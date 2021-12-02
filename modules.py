import os, json
import torch
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import pandas as pd

from transformers import (
    BertTokenizer, 
    VisualBertModel, 
    VisualBertConfig, 
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.models.visual_bert.modeling_visual_bert import VisualBertLMPredictionHead



from utils import load_tsv


class MMRad(pl.LightningModule):
  
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        super().__init__()

        self.args = args
        # TODO: Access from args / elsewhere:
        self.config = VisualBertConfig(visual_embedding_dim=2048)
        # Extracted features dim
        self.visual_features_dim = 1024

        self.train_size = train_size
        if args.load_model is None:
            print(f"Initialising Tx backbone from scratch\n")
            self.model = VisualBertModel(self.config)
        else:
            print(f"Loading transformer backbone from {args.load_model}\n")
            self.model = VisualBertModel(self.config).from_pretrained(args.load_model)

        if args.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.__init_tokenizer(tok=tokenizer)
        self.__init_transforms()

        # All pretext data aug tasks contained here
        self.pp = PretextProcessor(self.tokenizer)

    def __init_transforms(self):
        """Linear transform of input embeddings (from obj. detector
        to input of Tx encoder."""
        # TODO: fix up the 
        self.transform_img_ft = nn.Linear(self.visual_features_dim, self.config.visual_embedding_dim)
        self.transform_img_box = nn.Linear(4, self.config.visual_embedding_dim)
        self.transform_ln_ft = nn.LayerNorm(self.config.visual_embedding_dim)
        self.transform_ln_box = nn.LayerNorm(self.config.visual_embedding_dim)

    def __init_tokenizer(self, tok):
        
        if os.path.exists('./'+tok+'/'):
            tok_path = './'+tok+'/'
            print("Local Tokenizer exists")
        else:
            # download
            tok_path = tok
        self.tokenizer = BertTokenizer.from_pretrained(
            tok_path,
            do_lower_case=True
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # Preprocess based on the pretext tasks
        loss, acc = self.shared_step(batch, batch_idx)
        # result = pl.TrainResult(loss)

        logs = {'train_loss': loss, 'train_acc': acc}

        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = self.args.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        # result = pl.EvalResult(checkpoint_on = loss)

        logs = {'val_loss': loss, 'val_acc': acc}        
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = self.args.batch_size)

        return loss

    def shared_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        
        steps_per_epoch = self.train_size // self.args.batch_size
        total_epochs = self.args.epochs

        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)

        linear_warmup = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_epochs*steps_per_epoch*self.args.warmup_ratio), 
            num_training_steps=total_epochs*steps_per_epoch, 
            # last_epoch=self.current_epoch
            )
   
        lr_scheduler = {'scheduler':linear_warmup,
                        'name':'learning_rate',
                        'interval':'step',
                        'frequency':1}

        return [optimizer], [lr_scheduler]

class MMRadForPretraining(MMRad):
    
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        super().__init__(args, train_size, tokenizer=tokenizer)

        self.__init_pretraining_heads()

    def __init_pretraining_heads(self):
        self.text_prediction_head = VisualBertLMPredictionHead(self.config)
        self.seq_relationship = nn.Linear(self.config.hidden_size, 2)
        # self.image_mfr_prediction

    def shared_step(self, batch, batch_idx):

        # a batch should be a dict containing
        # input_ids, attention_mask, token_type_ids (for seq_prediction task), position_ids,
        # visual_embeds, visual_attention_mask, return_dict=True
    
        # process batch with pretext obj - MLM
        batch = self.pp.tokenize_pad_vectorize(batch)
        batch = self.pp.mask_txt(batch)

        # linear map img input to tx dim and add positions
        
        embed_ft = self.transform_ln_ft(self.transform_img_ft(batch['img']['features']))
        embed_pos =  self.transform_ln_box(self.transform_img_box(batch['img']['boxes']))
        visual_embeds = torch.div(torch.add(embed_ft, embed_pos), 2)

        # TODO: Handle elsewhere (preproc)
        num_features = batch['img']['num_boxes'][0]
        visual_attention_mask=torch.ones((len(batch['img']['id']), num_features)).to(self.device)
        visual_token_type_ids=torch.zeros((len(batch), num_features)).to(self.device)
       
        # just make labels = inputs (for now)
        # visual_labels = visual_embeds

        # labels = torch.hstack([batch['txt']['input_ids'], visual_labels])
        
        # MLM only for now
        
        labels = batch['txt']['masked_labels']

        #Dummy visual labels
        visual_labels = torch.full((labels.size()[0],36),-100, device=self.device)
        
        labels = torch.hstack((labels,visual_labels))

        outputs = self(
            input_ids=batch['txt']['masked_input_ids'],
            attention_mask=batch['txt']['att_mask'],
            # token_type_ids=batch['txt']['type_ids'],    # Let model auto-compute
            # position_ids=batch['txt']['pos_ids'],       # let model auto (use absolute pos)
            head_mask=None,
            inputs_embeds=None,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=None,
            image_text_alignment=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Run through PT heads - MLM only for now.
        # Most code borrowed from HF visualbertforpretraining
        sequence_output, pooled_output = outputs[:2]
        text_prediction_scores = self.text_prediction_head(sequence_output)
        # truncate to text ouput only
        # text_logits = text_prediction_scores[:,:self.args.max_seq_len,:]
        text_logits = text_prediction_scores
        text_preds = text_logits[(labels > 0), :].argmax(1)
        filtered_labels = labels[(labels > 0)]

        # pooled_output = self.seq_relationship(pooled_output)
        total_loss = None
        acc = None
        if labels is not None:
            # TODO: Increase size for txt+image
            total_size = batch['txt']['att_mask'].size(-1) + visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(
                    f"The labels provided should have same sequence length as total attention mask. "
                    f"Found labels with sequence length {labels.size(-1)}, expected {total_size}."
                )

            loss_fct = CrossEntropyLoss()
            # total_loss = loss_fct(text_logits.contiguous().view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = loss_fct(text_logits.view(-1, self.config.vocab_size), labels.view(-1))
            acc = (text_preds == filtered_labels).type(torch.float).mean()*100
        
        return total_loss,acc

class MMRadForClassification(MMRad):
    """Adds head for image classification"""
    def __init__(self, args, train_size, num_classes, tokenizer='bert-base-uncased'):
        super().__init__(args, train_size, tokenizer=tokenizer)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cls = nn.Linear(self.config.hidden_size, num_classes)
        # TODO: put this into args, make specific args for class
        self.img_only = False
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        logs = {'train_loss': loss, 'train_acc': acc}

        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = self.args.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)

        logs = {'val_loss': loss, 'val_acc': acc}        
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = self.args.batch_size)

        return loss
    def shared_step(self, batch, batch_idx):
        # a batch should be a dict containing:
        #   - input_ids
        #   - attention_mask
        #   - token_type_ids (for seq_prediction task)
        #   - position_ids
        #   - visual_embeds
        #   - visual_attention_mask
    
        # process txt
        if not self.img_only:
            batch = self.pp.tokenize_pad_vectorize(batch)
            batch['txt']['input_ids'] = batch['txt']['input_ids'].to(self.device)
        else:
            # TODO: device calls
            batch['txt']['input_ids'] = torch.zeros(len(batch), self.max_seq_len).to(self.device)
            batch['txt']['att_mask'] = torch.zeros(len(batch), self.max_seq_len).to(self.device)
            batch['txt']['type_ids'] = torch.zeros(len(batch), self.max_seq_len).to(self.device)
            batch['txt']['pos_ids'] = torch.ones_like(batch['txt']['input_ids']).to(self.device)
            batch['txt']['pos_ids'] *= torch.arange(0,batch['txt']['input_ids'].size()[1], 1).to(self.device)
        
        # linear map img input to tx dim and add positions
        embed_ft = self.transform_ln_ft(self.transform_img_ft(batch['img']['features']))
        embed_pos =  self.transform_ln_box(self.transform_img_box(batch['img']['boxes']))
        visual_embeds = torch.div(torch.add(embed_ft, embed_pos), 2)

        # TODO: Handle elsewhere (preproc)
        num_features = batch['img']['num_boxes'][0]
        visual_attention_mask=torch.ones((len(batch['img']['id']), num_features)).to(self.device)
       
        labels = batch['label']

        outputs = self(
            input_ids=batch['txt']['input_ids'],
            attention_mask=batch['txt']['att_mask'],
            # token_type_ids=batch['txt']['type_ids'],    # Let model auto-compute
            # position_ids=batch['txt']['pos_ids'],       # let model auto (use absolute pos)
            head_mask=None,
            inputs_embeds=None,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=None,
            image_text_alignment=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Run through PT heads - MLM only for now.
        # Most code borrowed from HF visualbertforpretraining
        # sequence_output.shape = (batch_size, max_seq_len, hidden_dim)
        # pooled_output.shape = (batch_size, 768)
        sequence_output, pooled_output = outputs[:2]
        cls_scores = self.cls(pooled_output)
        preds = cls_scores.argmax(1)

        total_loss = None
        acc = None

        loss_fct = CrossEntropyLoss()
        total_loss = loss_fct(cls_scores, labels)
        acc = (preds == labels).type(torch.float).mean()*100
    
        return total_loss,acc

class PretextProcessor:
    """Contains methods to mask etc"""
    def __init__(self, tokenizer, max_seq_len=20, mlm_rate=0.15):
        self.mlm_rate = mlm_rate
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def tokenize_pad_vectorize(self, batch):
        # batch['txt']['tokens'] = [self.tok.convert_tokens_to_ids(['[CLS]']+self.tok.tokenize(s.strip())+['[SEP]'])
        #                         for s in batch['txt']['raw']]
        # pad_fct = lambda a,i: a[0:i] if len(a) > i else a + [0]*(i-len(a))
        # TODO: .batch_encode_plus
        encoded = [self.tok.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length = self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        ) for sent in batch['txt']['raw']]

        # TODO: fixup device calls
        # for now, leave input ids on cpu for random choice (fix)
        batch['txt']['input_ids'] = torch.vstack([e['input_ids'] for e in encoded])
        batch['txt']['att_mask'] = torch.vstack([e['attention_mask'] for e in encoded]).to(self.device)
        # batch['txt']['input_ids'] = torch.tensor([pad_fct(sent,self.max_seq_len) for sent in batch['txt']['tokens']],
        #                                     dtype=torch.int)  
        
        # Generate other needed inputs for vbert/tx models
        batch['txt']['type_ids'] = torch.zeros_like(batch['txt']['input_ids']).to(self.device)
        
        batch['txt']['pos_ids'] = torch.ones_like(batch['txt']['input_ids']).to(self.device)
        batch['txt']['pos_ids'] *= torch.arange(0,batch['txt']['input_ids'].size()[1], 1).to(self.device)

        return batch
    
    def mask_txt(self, batch):
        """Returns masked inputs and labels over text inputs
        Generally follows https://keras.io/examples/nlp/masked_language_modeling/"""
        inp_mask = torch.rand(batch['txt']['input_ids'].size(), dtype=torch.float32) < self.mlm_rate

        # Avoid masking CLS(101), SEP(102) and padded (0)
        avoid_mask = torch.add(batch['txt']['input_ids']==0,
                               torch.add(batch['txt']['input_ids']==101,
                                         batch['txt']['input_ids']==102))
        

        inp_mask[avoid_mask] = False
        
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['txt']['input_ids'])
        labels[inp_mask] = batch['txt']['input_ids'][inp_mask]
        
        # Mask inputs
        masked_input_ids = batch['txt']['input_ids'].detach().clone()       
        # '[MASK]' is 103
        # of .15, 0.1 remain unchanged
        inp_mask_2m = inp_mask & (torch.rand(batch['txt']['input_ids'].size(), dtype=torch.float32) < 0.9)

        masked_input_ids[inp_mask_2m] = self.tok.convert_tokens_to_ids('[MASK]')

        # and 0.1 to random from the batch
        inp_mask_2r = inp_mask_2m & (torch.rand(batch['txt']['input_ids'].size(), dtype=torch.float32) < 1/9)
        masked_input_ids[inp_mask_2r] = np.random.choice(batch['txt']['input_ids'][~avoid_mask])
        
        batch['txt']['masked_input_ids'] = masked_input_ids.to(self.device)
        batch['txt']['masked_labels'] = labels.to(self.device)

        # TODO: can remove once handled this better
        batch['txt']['input_ids'] = batch['txt']['input_ids'].to(self.device)
        # TODO: Correct format?
        return batch

    def itm_sampling(self, batch):
        """Get negative samples and set is_matched labels
        for the ITM task"""
        pass

class MMRadDM(pl.LightningDataModule):
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("MMRadDM")
    #     # Data splits
    #     parser.add_argument("--train", dest='train_split', default='mscoco_train')
    #     parser.add_argument("--valid", dest='valid_split', default='mscoco_val')
    #     parser.add_argument("--test", default=None)
    #     parser.add_argument("--dropLast", dest='drop_last', default=True)
    #     parser.add_argument("--shuffle", default=True)
    #     parser.add_argument("--topk", default=5120)
    #     parser.add_argument("--topkVal", dest='val_topk', default=None)

    #     # Sizing
    #     parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    #     parser.add_argument('--batchSizeVal', dest='valid_batch_size', type=int, default=256)

    #     # Data path
    #     parser.add_argument('--dataPath', dest='data_path', default="/media/matt/data21/mmRad/")

    #     return parent_parser
    
    
    def __init__(self, args):
        super().__init__()

        # Add args
        self.batch_size=args.batch_size
        # TODO: fix so can have other sizes
        self.valid_batch_size=args.valid_batch_size 
        self.shuffle=args.shuffle
        self.drop_last = args.drop_last
        self.shuffle=args.shuffle
        self.topk=args.topk
        self.val_topk=args.val_topk
        self.train_split=args.train_split
        self.valid_split=args.valid_split
        self.data_path=args.data_path

        self.num_workers = os.cpu_count()

    def prepare_data(self):
        # Called on 1 GPU only
        pass

    def setup(self, stage=None):
        # Called on every GPU
        if stage=='fit' or stage is None:
            
            if self.train_split=='mscoco_train':
                self.train_dset = CocoDataset(self.data_path+'captions_train2017.json',
                        self.data_path+'img_features/mscoco-train_2017-custom.tsv', topk=self.topk)
                self.train_size = len(self.train_dset)
            # Will always val on coco if train on coco
            # if self.valid_split=='mscoco_val':
                self.valid_dset = CocoDataset(self.data_path+'captions_val2017.json',
                        self.data_path+'img_features/mscoco-val_2017-custom.tsv', topk=self.val_topk)
                self.valid_size = len(self.valid_dset)
            
            elif self.train_split=='cub_train':
                # CUB dset is not split into train/val
                cubdata = CubDataset(self.data_path+'CUB/caption_label_data.csv',
                        self.data_path+'CUB/cub_all.tsv', topk=self.topk)
                train_set_size = int(len(cubdata)*0.8)
                valid_set_size = len(cubdata) - train_set_size
                self.train_dset, self.valid_dset = random_split(cubdata, [train_set_size, valid_set_size])
                self.train_size = len(self.train_dset)

                # Store number of classes
                self.num_classes = cubdata.get_num_classes()




        if stage=='test' or stage is None:
            pass

        # self.dims = len(self.train_dset)//self.batch_size

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dset, batch_size=self.batch_size,
            shuffle=self.shuffle,
            # collate_fn=lambda x: x,
            drop_last=self.drop_last, pin_memory=True,
            num_workers=self.num_workers
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_dset, batch_size=self.valid_batch_size,
            shuffle=False,
            # collate_fn=lambda x: x,
            drop_last=False, pin_memory=True,
            num_workers=self.num_workers
        )
        return dl    

# class MyCallBack(Callback):
#     def on_epoch_start(self, trainer, pl_module):
#         print("\n")
# class InputMonitor(Callback):
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#         if (batch_idx + 1) % trainer.log_every_n_steps == 0:
#             x, y = batch
#             logger = trainer.logger
#             logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
#             logger.experiment.add_histogram("target", y, global_step=trainer.global_step)

class CubDataset(Dataset):
    """CUB images + captions (1 each) for fine tuning
    & eval. csv_path is location to csv built from join_cub_csv()"""
    def __init__(self, csv_path, img_ft_path, topk=None):
        super().__init__()
        self.txt_df = pd.read_csv(csv_path)
        # Loads id, height, width, num_boxes, boxes, features
        self.img_data = load_tsv(img_ft_path, topk=topk)
        self.txt_data = [{'img_id':item['id'], 
                          'caption':item['captions'],
                          'label':item['class']} 
                          for _,item in self.txt_df.iterrows()]
        if topk != None:
            # Filter img_ids to match loaded topk
            self.txt_data = self.txt_data[:topk]

    def __len__(self):
        return len(self.img_data)
    
    def get_num_classes(self):
        return len(set(self.txt_df['class'].astype(int)))
    
    def __getitem__(self, idx):
        # Re-using from coco - not efficient

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Produces a sample per txt sequence- image features are duplicated for each.        
        img_id = str(self.txt_data[idx]['img_id'])
        caption = self.txt_data[idx]['caption']
        label = int(self.txt_data[idx]['label'])
        img_data = self.img_data[img_id]
        # Create nested
        sample = {'txt': {'raw' : caption}, 
                  'img': {'id' : img_id, 
                          'features' : img_data['features'], 
                          'boxes' : img_data['boxes'],
                          'num_boxes' : img_data['num_boxes'], 
                          'img_h' : img_data['img_h'],
                          'img_w' : img_data['img_w']
                          },
                  'label': label
                 }
        return sample

class CocoDataset(Dataset):
    """MS-COCO dataset captions only
    No transforms/process here"""
    def __init__(self, json_fp, img_ft_path, topk=5120):
        super().__init__()

        with open(json_fp) as f:
            self.metadata = json.load(f)
        # Dict of image fts' by id
        self.img_data = load_tsv(img_ft_path, topk=topk)
        # Add captions as duplicate tuples
        self.txt_data = [{'img_id':item['image_id'], 'caption':item['caption']} 
                          for item in self.metadata['annotations']]
        if topk != None:
            # Filter img_ids to match loaded topk
            self.txt_data = [item for item in self.txt_data
                             if self.img_data.get(str(item['img_id']), 0) != 0]

    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Produces a sample per txt sequence- image features are duplicated for each.        
        img_id = str(self.txt_data[idx]['img_id'])
        caption = self.txt_data[idx]['caption']
        img_data = self.img_data[img_id]
        # Create nested
        sample = {'txt': {'raw' : caption}, 
                  'img': {'id' : img_id, 
                          'features' : img_data['features'], 
                          'boxes' : img_data['boxes'],
                          'num_boxes' : img_data['num_boxes'], 
                          'img_h' : img_data['img_h'],
                          'img_w' : img_data['img_w']
                          }
                 }
        return sample


