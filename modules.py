import os, json, random
import torch, torchmetrics
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, GELU, BCELoss
from torch.nn.modules.normalization import LayerNorm
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import pandas as pd
from ast import literal_eval
from sklearn.metrics import roc_auc_score

from transformers import (
    BertTokenizer, 
    VisualBertModel, 
    VisualBertConfig, 
    AdamW,
    get_linear_schedule_with_warmup,
    BertLayerNorm
)
from transformers.models.visual_bert.modeling_visual_bert import VisualBertLMPredictionHead



from utils import load_tsv





class MMRad(pl.LightningModule):
  
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        super().__init__()
        self.save_hyperparameters(args)

        self.config = VisualBertConfig(visual_embedding_dim=self.hparams.visual_embedding_dim,
                                       num_attention_heads=self.hparams.num_attention_heads,
                                       dropout=self.hparams.dropout,
                                       hidden_size=self.hparams.encoder_hidden_size,
                                       num_hidden_layers=self.hparams.num_tx_layers)
        # Extracted features
        self.visual_features_dim = 1024

        self.train_size = train_size
        
        if self.hparams.load_model is None:
            print(f"Initialising Tx backbone from scratch\n")
            self.model = VisualBertModel(self.config)
            self.model.apply(self.init_weights)
        else:
            model_path = self.hparams.load_model
            print(f"Loading transformer backbone from {model_path}\n")
            self.model = VisualBertModel(self.config).from_pretrained(model_path)

        if self.hparams.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._init_tokenizer(tok=tokenizer)
        self._init_transforms()
        # All pretext data aug tasks contained here
        self.pp = PretextProcessor(self.tokenizer, max_seq_len=self.hparams.max_seq_len)

    def _init_transforms(self):
        """Linear transform of input embeddings (from obj. detector
        to input of Tx encoder."""
        # self.config.visual_embedding_dim gets overidden by a preloaded model.
        # So use this instead
        in_features_dim = self.model.embeddings.visual_projection.in_features
        self.transform_img_ft = nn.Sequential(
            nn.Linear(self.visual_features_dim, in_features_dim),
            nn.LayerNorm(in_features_dim)
        )
        self.transform_img_box = nn.Sequential(
            nn.Linear(4, in_features_dim),
            nn.LayerNorm(in_features_dim)
       )
        self.transform_img_box.apply(self.init_weights)
        self.transform_img_ft.apply(self.init_weights)
    
    def vis_pos_embeds(self, img_ft, img_box):
        
        embed_ft = self.transform_img_ft(img_ft)
        embed_pos = self.transform_img_box(img_box)
        return torch.div(torch.add(embed_ft, embed_pos), 2)
    
    def _init_tokenizer(self, tok):
        
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
        print(f"Using tokenizer: {self.hparams.tokenizer}")

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, **inputs):
        return self.model(**inputs)

   
    def configure_optimizers(self):
        
        steps_per_epoch = self.train_size // self.hparams.batch_size
        total_epochs = self.hparams.epochs

        # optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        optimizer = AdamW(self.parameters(), 
                          lr=self.hparams.lr, 
                          weight_decay=self.hparams.weight_decay)

        linear_warmup = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_epochs*steps_per_epoch*self.hparams.warmup_ratio), 
            num_training_steps=total_epochs*steps_per_epoch, 
            # last_epoch=self.current_epoch
            )
   
        lr_scheduler = {'scheduler':linear_warmup,
                        'name':'learning_rate',
                        'interval':'step',
                        'frequency':1}
        if self.hparams.lr_scheduler:
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

class MMRadForPretraining(MMRad):
    
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        super().__init__(args, train_size, tokenizer=tokenizer)

        self.__init_pretraining_heads()
        self.task_step = {'mlm':self.mlm_step, 'mfr':self.mfr_step, 'itm':self.itm_step,
                          'wwm':self.wwm_step}
        self.hparams.tasks = literal_eval(self.hparams.tasks)


    def __init_pretraining_heads(self):
        self.text_prediction_head = VisualBertLMPredictionHead(self.config)
        self.seq_relationship_head = nn.Linear(self.config.hidden_size, 2)
        self.image_mfr_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.visual_features_dim),
                GELU(),
                LayerNorm(self.visual_features_dim, eps=1e-12)
                )
    def wwm_step(self, batch, batch_idx):
        return self.mlm_step(batch, batch_idx, subtask='wwm')

    def mlm_step(self, batch, batch_idx, subtask='mlm'):
        if subtask=='mlm':
            batch = self.pp.mask_txt(batch)
        elif subtask=='wwm':
            batch = self.pp.whole_word_mask(batch)

        # TODO: Fix this up so it can be called in shared step (see mfr_step)
        batch = self.pp.img_vectorize(batch, model=self)     

        txt_labels = batch['txt']['masked_labels']
        #Dummy visual labels
        img_labels = torch.full((txt_labels.size()[0],36),-100, device=self.device)
        
        # labels = torch.hstack((txt_labels,img_labels))

        outputs = self(
            input_ids=batch['txt']['masked_input_ids'],
            attention_mask=batch['txt']['att_mask'],
            # token_type_ids=batch['txt']['type_ids'],    # Let model auto-compute
            # position_ids=batch['txt']['pos_ids'],       # let model auto (use absolute pos)
            head_mask=None,
            inputs_embeds=None,
            visual_embeds=batch['img']['visual_embeds'],
            visual_attention_mask=batch['img']['att_mask'],
            visual_token_type_ids=None,
            image_text_alignment=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Most code borrowed from HF visualbertforpretraining
        sequence_output, pooled_output = outputs[:2]
        txt_sequence, img_sequence = torch.split(sequence_output, [txt_labels.shape[1], img_labels.shape[1]], dim=1)

        text_logits = self.text_prediction_head(txt_sequence)

        # text ouput only
        text_preds = text_logits[(txt_labels > 0), :].argmax(1)
        filtered_labels = txt_labels[(txt_labels > 0)]

        loss_fct = CrossEntropyLoss()
        # total_loss = loss_fct(text_logits.contiguous().view(-1, self.config.vocab_size), labels.view(-1))
        loss = loss_fct(text_logits.view(-1, self.config.vocab_size), txt_labels.view(-1))
        acc = (text_preds == filtered_labels).type(torch.float).mean()*100
        return {'loss':loss, 'acc':acc}

    def mfr_step(self, batch, batch_idx):
        # num_features = batch['img']['num_boxes'][0]
        # batch_size = self.hparams.batch_size

        # Mask before projection
        batch = self.pp.mask_img(batch) 
        batch = self.pp.img_vectorize(batch, model=self)     
        #Dummy text labels
        txt_labels = torch.full_like(batch['txt']['input_ids'], -100, device=self.device)
        
        label_mask = batch['img']['label_mask']
        # update labels with features that were masked
        img_labels = batch['img']['features']

        outputs = self(
            input_ids=batch['txt']['input_ids'],
            attention_mask=batch['txt']['att_mask'],
            # token_type_ids=batch['txt']['type_ids'],    # Let model auto-compute
            # position_ids=batch['txt']['pos_ids'],       # let model auto (use absolute pos)
            head_mask=None,
            inputs_embeds=None,
            visual_embeds=batch['img']['visual_embeds'],
            visual_attention_mask=batch['img']['att_mask'],
            visual_token_type_ids=None,
            image_text_alignment=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Most code borrowed from HF visualbertforpretraining
        sequence_output, pooled_output = outputs[:2]
        txt_sequence, img_sequence = torch.split(sequence_output, [txt_labels.shape[1], img_labels.shape[1]], dim=1)

        img_projected = self.image_mfr_head(img_sequence)
        loss_fct = MSELoss()
        
        loss = loss_fct(img_projected[label_mask], img_labels[label_mask])
        # eps = 0.5
        # acc = 100*torch.sum(torch.sum( (torch.mean(img_projected[label_mask], dim=1) - 
        #                  torch.mean(img_labels[label_mask], dim=1)) < eps ))/batch_size
        
        return {'loss':loss}

    def itm_step(self, batch, batch_idx):

        batch = self.pp.itm_sampling(batch)        
        batch = self.pp.img_vectorize(batch, model=self)     
        
        outputs = self(
            input_ids=batch['txt']['input_ids'],
            attention_mask=batch['txt']['att_mask'],
            # token_type_ids=batch['txt']['type_ids'],    # Let model auto-compute
            # position_ids=batch['txt']['pos_ids'],       # let model auto (use absolute pos)
            head_mask=None,
            inputs_embeds=None,
            visual_embeds=batch['img']['visual_embeds'],
            visual_attention_mask=batch['img']['att_mask'],
            visual_token_type_ids=None,
            image_text_alignment=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = self.seq_relationship_head(pooled_output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(seq_relationship_score.view(-1,2), batch['is_matched'].view(-1))
        acc = (batch['is_matched'].view(-1) == seq_relationship_score.argmax(1).view(-1)).type(torch.float).mean()*100
        return {'loss':loss, 'acc':acc}             
        
    def training_step(self, batch, batch_idx):
        # Sample a pretext task for each iteration
        task = random.choice(self.hparams.tasks)

        # Preprocess based on the pretext tasks
        metrics = self.shared_step(batch, batch_idx, task)
        logs = {'train_'+task+'_'+k:v for k,v in metrics.items()}
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True,
                      logger = True, batch_size = self.hparams.batch_size)
        return metrics['loss']
    
    def validation_step(self, batch, batch_idx):
        tot_loss = 0
        logs = {}
        # Validation step calculates loss for all selected tasks
        for task in self.hparams.tasks:
            metrics = self.shared_step(batch, batch_idx, task)
            tot_loss += metrics['loss']

            task_log = {'val_'+task+'_'+k:v for k,v in metrics.items()}
            logs.update(task_log)
            
        logs['val_avg_loss'] = tot_loss/len(self.hparams.tasks)
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, 
                      logger = True, batch_size = self.hparams.valid_batch_size)
        return logs['val_avg_loss']

    def shared_step(self, batch, batch_idx, task):

        # a batch should be a dict containing
        # input_ids, attention_mask, token_type_ids (for seq_prediction task), position_ids,
        # visual_embeds, visual_attention_mask, return_dict=True
    
        # process batch: txt- tokenize/pad, img: project input to Tx dims
        batch = self.pp.tokenize_pad_vectorize(batch)

        # Run sampled task, return results as a dict
        return self.task_step[task](batch, batch_idx)

class MMRadForClassification(MMRad):
    """Adds head for image classification"""
    def __init__(self, args, train_size, n_classes, labelset=None, n_hidden=512, 
                 tokenizer='bert-base-uncased'):
        super().__init__(args, train_size, tokenizer=tokenizer)
        
        self.labelset = labelset
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cls = nn.Linear(self.config.hidden_size, n_classes)
        self.cls.apply(self.init_weights)

        if self.hparams.img_only:
            print(f"Setting text inputs to 0 / Masking")
        if self.hparams.txt_only:
            print(f"Txt only- masking visual elements")

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)

        logs = {'train_'+k:v for k,v in metrics.items() if k!='preds'}
        self.log_dict(logs, on_step = False, on_epoch = True, 
                      prog_bar = True, logger = True, batch_size = self.hparams.batch_size)
        
        return metrics['loss']

        # logs = {'train_loss': loss}
        # for name,val in zip(self.mimic_labelset, acc):
        #     logs['train_'+name]=val
        # for name,val in zip(self.mimic_labelset, pos):
        #     logs['train_counts_'+name] = val
        # # logs = {'train_loss': loss, 'train_acc': acc}

        # self.log_dict(logs, on_step = False, on_epoch = True, prog_bar = True, logger = True, batch_size = self.hparams.batch_size)
        # return loss
    
    def validation_step(self, batch, batch_idx):

        metrics = self.shared_step(batch, batch_idx)
      
        # Log per-step metrics
        logs = {'val_'+k:v for k,v in metrics.items() if k!='preds'}
        self.log_dict(logs, on_step = False, on_epoch = True, 
                      prog_bar = True, logger = True, batch_size = self.hparams.valid_batch_size)
        return {'loss': metrics['loss'], 'preds':metrics['preds']}

    def shared_step(self, batch, batch_idx):
        # a batch should be a dict containing:
        #   - input_ids
        #   - attention_mask
        #   - token_type_ids (for seq_prediction task)
        #   - position_ids
        #   - visual_embeds
        #   - visual_attention_mask
    
        # process txt
        if self.hparams.img_only:
            batch_size = len(batch['img']['id'])
            seq_len = self.hparams.max_seq_len
            batch['txt']['input_ids'] = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.int)
            batch['txt']['att_mask'], batch['txt']['type_ids'] = batch['txt']['input_ids'], batch['txt']['input_ids']
            batch['txt']['pos_ids'] = torch.ones(batch_size, seq_len, device=self.device)
            batch['txt']['pos_ids'] *= torch.arange(0,seq_len, 1, device=self.device)

        else:
            batch = self.pp.tokenize_pad_vectorize(batch)
            batch['txt']['input_ids'] = batch['txt']['input_ids'].to(self.device)

            ## img input to tx dim and add positions
            visual_embeds = self.vis_pos_embeds(img_ft=batch['img']['features'],
                                                img_box=batch['img']['boxes'])
            num_features = batch['img']['num_boxes'][0]

            if self.hparams.txt_only:
                visual_attention_mask=torch.ones((len(batch['img']['id']), num_features)).to(self.device)
            else:
                visual_attention_mask=torch.zeros((len(batch['img']['id']), num_features)).to(self.device)

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

        # sequence_output.shape = (batch_size, max_seq_len, hidden_dim)
        # pooled_output.shape = (batch_size, 768)
        sequence_output, pooled_output = outputs[:2]
        
        # #### MIMIC class
          
        logits = self.cls(self.dropout(pooled_output))
        preds = nn.Sigmoid()(logits) 
        
        loss_fct = nn.BCEWithLogitsLoss()
        if self.hparams.easy_classification:
            logits = logits.view(-1)
            loss = loss_fct(logits, labels)
            acc = ((preds.view(-1) > 0.5) == labels).type(torch.float).mean()*100
            metrics = {'loss':loss, 'acc':acc}
            
        else:
            loss = loss_fct(logits, labels)
            acc = (preds == labels).type(torch.float).mean(dim=0)
            # positives = torch.sum(labels, dim=0)
            metrics = {'loss':loss}
            metrics['preds']=preds


        return metrics
        ## Old: CUB 
        # cls_scores = self.cls(pooled_output)
        # preds = cls_scores.argmax(1)

        # total_loss = None
        # acc = None

        # loss_fct = CrossEntropyLoss()
        # total_loss = loss_fct(cls_scores, labels)
        # acc = (preds == labels).type(torch.float).mean()*100

class LogValMetrics(pl.Callback):
    """Log whole-val auroc & TP,FP,TN,FP stats using accumulated preds & labels
       Will probably break on distributed GPU training.."""
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.auroc = torchmetrics.AUROC(num_classes=14,
                                        average=None)
        self.statscores = torchmetrics.StatScores(reduce='macro',
                                                  num_classes=14,
                                                    )
        self.auroc.to(self.device)
        self.statscores.to(self.device)

        self.result_auc = torch.zeros((14,), device=self.device)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx,_) -> None:
        
        # Accumulate preds/labels across batches
        if batch_idx==0:
            self.val_preds = outputs['preds']
            self.val_labels = batch['label']
        else:
            self.val_preds = torch.vstack((self.val_preds,outputs['preds']))
            self.val_labels = torch.vstack((self.val_labels, batch['label'])) 

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Skip labels that don't have both instances (0,1); no chance of all 1's
        mask = torch.sum(self.val_labels, dim=0) > 0
        self.auroc.num_classes = torch.sum(mask)

        # Compute & update AUC for the others; carry over old vals (0) 
        self.result_auc[mask] = self.auroc(self.val_preds[:,mask], self.val_labels[:,mask].type(torch.int)).to(self.device)

        # Compute stat scores (TP,...) over epoch
        # StatScores returns tensor of shape (num_classes, 5)
        # When macro is used. last dim is [TP,FP,TN,FN,TP+FN]
        statscores = self.statscores(self.val_preds, self.val_labels.type(torch.int)).type(torch.float)
        
        for name,score,stats in zip(pl_module.labelset, self.result_auc, torch.tensor_split(statscores,14,dim=0)):
            stats = stats.squeeze(0)

            results = {'AUC_'+name:score,
                       'SS_'+name+'_TP':stats[0],
                       'SS_'+name+'_FP':stats[1],
                       'SS_'+name+'_TN':stats[2],
                       'SS_'+name+'_FN':stats[3],
                       'SS_'+name+'_SUP':stats[4]}


            self.log_dict(results)

class CheckpointEveryNEpochs(pl.Callback):
    def __init__(self, dirpath,every_n_epochs):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs  
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (pl_module.current_epoch!=0) and (pl_module.current_epoch % self.every_n_epochs == 0):
            trainer.save_checkpoint(self.dirpath+"epoch_"+pl_module.current_epoch)
            

class PretextProcessor:
    """Contains preproc and pretext task methods"""
    def __init__(self, tokenizer, max_seq_len=20, mlm_rate=0.15, mfr_rate=0.15,
                 itm_rate=0.5):
        self.mlm_rate = mlm_rate
        self.mfr_rate = mfr_rate
        self.itm_rate = itm_rate

        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def img_vectorize(self, batch, model):
        '''Creates necessary visual inputs for vbert model'''
        num_features = batch['img']['num_boxes'][0]

        # batch['img']['masked_features'] = batch['img'].get('masked_features',batch['img']['features'])
        batch['img']['visual_embeds'] = model.vis_pos_embeds(
                # If masking, use those, if not, use raw input.
                img_ft=batch['img'].get('masked_features',batch['img']['features']),
                img_box=batch['img']['boxes']
                )
        batch['img']['att_mask'] = torch.ones((len(batch['img']['id']), num_features), device=self.device)
        batch['img']['type_ids'] = torch.zeros((len(batch), num_features), device=self.device)
        return batch

    def tokenize_pad_vectorize(self, batch):
       
        # transformers > 4.0.0 replace
        encoded = self.tok(
            text=batch['txt']['raw'],
            add_special_tokens=True,
            max_length = self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        ) 
        
        batch['txt']['input_ids'] = encoded['input_ids'].to(self.device)
        batch['txt']['att_mask'] = encoded['attention_mask'].to(self.device)

        
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
        
        # Mask inputs: of 0.15, 0.1 remain unchanged
        masked_input_ids = batch['txt']['input_ids'].detach().clone()       
        inp_mask_2m = inp_mask & (torch.rand(batch['txt']['input_ids'].size(), dtype=torch.float32) < 0.9)
        masked_input_ids[inp_mask_2m] = torch.full_like(  # '[MASK]' is 103
            masked_input_ids[inp_mask_2m],
            self.tok.convert_tokens_to_ids('[MASK]')
         ) 

        # and 0.1 to random from the batch
        inp_mask_2r = inp_mask_2m & (torch.rand(batch['txt']['input_ids'].size(), dtype=torch.float32) < 1/9)
 
        bs = len(batch['txt']['input_ids'])
        seq_len = len(batch['txt']['input_ids'][0])
        r_idx = torch.randint(0,bs*seq_len, (torch.sum(inp_mask_2r),))
        masked_input_ids[inp_mask_2r] = batch['txt']['input_ids'].view(-1)[r_idx]


        batch['txt']['masked_input_ids'] = masked_input_ids.to(self.device)
        batch['txt']['masked_labels'] = labels.to(self.device)

        return batch

    def mask_img(self, batch):
        """Returns batch with masked visual features and labels"""
        num_features = batch['img']['features'].shape[:2] # (256, 36)
        inp_mask = torch.rand(num_features, dtype=torch.float32) < self.mfr_rate
        
        masked_features = batch['img']['features'].detach().clone()
        # 0.1 remain unchanged
        inp_mask_2m = inp_mask & (torch.rand(num_features, dtype=torch.float32) < 0.9)
        # TODO: not sure why assigning 0 doesn't work
        masked_features[inp_mask_2m, :] = torch.zeros_like(masked_features[inp_mask_2m, :])
        # 0.1 to random feat
        inp_mask_2r = inp_mask_2m & (torch.rand(num_features, dtype=torch.float32) < 1/9)
        # gen sample for each of the masked; max idx is 256*36-1 (e.g.)
        r_idx = torch.randint(0,num_features[0]*num_features[1], (torch.sum(inp_mask_2r),))
        masked_features[inp_mask_2r,:] = batch['img']['features'].view(-1,batch['img']['features'].shape[2])[r_idx]

        batch['img']['masked_features'] = masked_features.to(self.device)
        batch['img']['label_mask'] = inp_mask

        return batch

    def whole_word_mask(self,batch):
        """Returns masked inputs and labels over text inputs
        samples from candidate whole words not parts of.
        batch: training data
        returns: batch w/ masked whole words.
        Processes on per-example basis, may be slow.
        Roughly follows https://github.com/huggingface/transformers/blob/07708793f20ec3a949ccab32cc4fe0c7272dcc4c/src/transformers/data/data_collator.py#L301"""
        # TODO: Vectorise this function

        # Instantiate masked inputs
        batch['txt']['masked_input_ids'] = batch['txt']['input_ids'].detach().clone()
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['txt']['input_ids'], device=self.device)        

        for sample_idx,sample_input_ids in enumerate(batch['txt']['input_ids']):

            input_tokens = self.tok.convert_ids_to_tokens(sample_input_ids)

            cand_indexes = []
            for (i, token) in enumerate(input_tokens):
                sent_len = 0
                if token == "[PAD]":
                    sent_len = i
                    break
                if token == "[CLS]" or token == "[SEP]":
                    continue
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])       
            
            random.shuffle(cand_indexes)
            num_to_predict = min(self.max_seq_len, max(1, int(round(sent_len * self.mlm_rate))))
            masked_lms = []
            covered_indexes = set()
            for index_set in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue
                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break
                if is_any_index_covered:
                    continue
                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)

            assert len(covered_indexes) == len(masked_lms)
            covered_indexes = list(covered_indexes)
            # mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

            batch['txt']['masked_input_ids'][sample_idx,covered_indexes] = torch.full((len(covered_indexes),),103, device=self.device)
            labels[sample_idx,covered_indexes] = sample_input_ids[covered_indexes]  
        batch['txt']['masked_labels'] = labels.to(self.device)
        return batch

    def itm_sampling(self, batch):
        """Get negative samples and set is_matched labels
        for the ITM task"""
        batch_size = len(batch['img']['id'])
        
        inp_mask = torch.rand(batch_size, dtype=torch.float32, device=self.device) < self.itm_rate

        # Generate random idx 
        rand_idx = torch.randperm(batch_size)[:sum(inp_mask)]

        ## Replace with negative samples
        batch['img']['features'][inp_mask] = batch['img']['features'][rand_idx]
        batch['img']['boxes'][inp_mask] = batch['img']['boxes'][rand_idx]

        # expand dim to match the seq_rel linear layer output
        batch['is_matched'] = torch.unsqueeze(~inp_mask, 1).long()
        
        return batch

class MMRadDM(pl.LightningDataModule):
    def __init__(self, args):
        
        super().__init__()

        self.save_hyperparameters(args)

        self.num_workers = os.cpu_count()
        self.g = torch.Generator()
        self.g.manual_seed(808)
    def prepare_data(self):
        # Called on 1 GPU only
        pass

    def setup(self, stage=None):
        # Called on every GPU
        if stage=='fit' or stage is None:
            

            
            if self.hparams.train_split=='mscoco_train':
                self.train_dset = CocoDataset(self.hparams.data_path+'captions_train2017.json',
                        self.hparams.data_path+'img_features/mscoco-train_2017-custom.tsv', topk=self.hparams.topk)
                self.valid_dset = CocoDataset(self.hparams.data_path+'captions_val2017.json',
                        self.hparams.data_path+'img_features/mscoco-val_2017-custom.tsv', topk=self.hparams.val_topk)
            
            elif self.hparams.train_split=='cub_train':
                # CUB dset is not split into train/val
                cubdata = CubDataset(self.hparams.data_path+'CUB/caption_label_data.csv',
                        self.hparams.data_path+'CUB/cub_all.tsv', topk=self.hparams.topk)
                train_set_size = int(len(cubdata)*0.9)
                valid_set_size = len(cubdata) - train_set_size
                self.train_dset, self.valid_dset = random_split(cubdata, [train_set_size, valid_set_size],
                                                                generator=self.g)
                # Store number of classes
                self.num_classes = cubdata.get_num_classes()

            elif self.hparams.train_split=='mimic_train':
                # MIMIC set not split into train/val
                #txt_path, img_path, labels_path, use_captions='findings', topk=5120
                mimic_root = os.path.join(self.hparams.data_path, 'MIMIC')
                txt_path = os.path.join(mimic_root, 'id_to_findings.csv')
                img_path = os.path.join(mimic_root, 'mimic_PT_AP-view.tsv')
                label_path = os.path.join(mimic_root, 'labels', 'mimic-cxr-2.0.0-chexpert.csv.gz')
                
                mimic_data = MimicDataset(txt_path, img_path, label_path,
                                          use_captions='findings', topk=self.hparams.topk,
                                          binary_task=self.hparams.easy_classification)
                train_set_size = int(len(mimic_data)*0.9)
                valid_set_size = len(mimic_data) - train_set_size
                self.train_dset, self.valid_dset = random_split(mimic_data, [train_set_size, valid_set_size],
                                                                generator=self.g)
                self.num_classes = 1 if self.hparams.easy_classification else 14
                self.labelset = mimic_data.labelset
            
            elif self.hparams.train_split=='mimic_ft':
                # MIMIC set not split into train/val
                #txt_path, img_path, labels_path, use_captions='findings', topk=5120
                mimic_root = os.path.join(self.hparams.data_path, 'MIMIC')
                txt_path = os.path.join(mimic_root, 'id_to_findings.csv')
                img_path = os.path.join(mimic_root, 'mimic_FT_AP-view.tsv')
                label_path = os.path.join(mimic_root, 'labels', 'mimic-cxr-2.0.0-chexpert.csv.gz')
                
                mimic_data = MimicDataset(txt_path, img_path, label_path,
                                          use_captions='findings', topk=self.hparams.topk,
                                          binary_task=self.hparams.easy_classification)
                train_set_size = int(len(mimic_data)*0.8)
                valid_set_size = len(mimic_data) - train_set_size
                self.train_dset, self.valid_dset = random_split(mimic_data, [train_set_size, valid_set_size],
                                                                generator=self.g)
                self.num_classes = 1 if self.hparams.easy_classification else 14
                self.labelset = mimic_data.labelset

            elif self.hparams.train_split=='mimic_all_findings':
                # MIMIC set not split into train/val
                #txt_path, img_path, labels_path, use_captions='findings', topk=5120
                mimic_root = os.path.join(self.hparams.data_path, 'MIMIC')
                txt_path = os.path.join(mimic_root, 'id_to_findings.csv')
                img_path = os.path.join(mimic_root, 'mimic_img_with_findings.tsv')
                label_path = os.path.join(mimic_root, 'labels', 'mimic-cxr-2.0.0-chexpert.csv.gz')
                
                mimic_data = MimicDataset(txt_path, img_path, label_path,
                                          use_captions='findings', topk=self.hparams.topk,
                                          binary_task=self.hparams.easy_classification)
                train_set_size = int(len(mimic_data)*0.8)
                valid_set_size = len(mimic_data) - train_set_size
                self.train_dset, self.valid_dset = random_split(mimic_data, [train_set_size, valid_set_size],
                                                                generator=self.g)
                self.num_classes = 1 if self.hparams.easy_classification else 14
                self.labelset = mimic_data.labelset                
        self.train_size,self.valid_size = len(self.train_dset), len(self.valid_dset)
        print(f"Size of train / val splits: {self.train_size} / {self.valid_size}")

        if stage=='test' or stage is None:
            pass

        # self.dims = len(self.train_dset)//self.batch_size
    def seed_worker(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def train_dataloader(self):
        dl = DataLoader(
            self.train_dset, batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            # collate_fn=debug_collate, #lambda x: x,
            drop_last=self.hparams.drop_last, pin_memory=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_dset, batch_size=self.hparams.valid_batch_size,
            shuffle=False,
            # collate_fn=debug_collate, #lambda x: x,
            drop_last=False, pin_memory=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        return dl    



class CubDataset(Dataset):
    """CUB images + captions (1 each) for fine tuning
    & eval. csv_path is location to csv built from join_cub_csv()"""
    def __init__(self, csv_path, img_ft_path, topk=0):
        super().__init__()
        self.txt_df = pd.read_csv(csv_path)
        # Loads id, height, width, num_boxes, boxes, features
        self.img_data = load_tsv(img_ft_path, topk=topk)
        self.txt_data = [{'img_id':item['id'], 
                          'caption':item['captions'],
                          'label':item['class']} 
                          for _,item in self.txt_df.iterrows()]
    
        if topk != 0:
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

class MimicDataset(Dataset):
    """Mimic-cxr dataset with extracted visual features,
    captions (from impressions), ID, view, ..."""
    def __init__(self, txt_path, img_path, labels_path, 
                 use_captions='findings', topk=0,
                 binary_task=False):
        super().__init__()
        self.binary_task = binary_task
        self.use_captions = use_captions
        
        self.img_data = load_tsv(img_path, topk=topk)
        self.txt_df = pd.read_csv(txt_path)
        self.labelset = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                         'Edema', 'Enlarged Cardiomedastinum', 'Fracture',
                         'Lung Lesion', 'Lung Opacity', 'No Finding',
                         'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                         'Pneumothorax', 'Support Devices']

        # Get label data, default chexpert
        self.label_data = pd.read_csv(labels_path)
        label_cats = self.label_data.columns[2:]
        

        # Filter img_ids to match loaded topk
        # This also filters on the PT or FT split of img features
        self.txt_df = self.txt_df[self.txt_df['dicom_id'].isin(self.img_data.keys())]

        # add labels - filter/sort on reports
        self.txt_df = self.txt_df.merge(self.label_data, on='study_id', how='left')
        self.labels_data = np.nan_to_num(np.asarray(self.txt_df[label_cats]))
        self.labels_data[self.labels_data < 0] = 0
        if self.binary_task:
            # 'No finding' label only
            self.labels_data = self.labels_data[:,8]
        self.txt_data = self.txt_df
        
    def __len__(self):
        return len(self.img_data)        
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Produces a sample per txt sequence- image features are duplicated for each.        
        selected = self.txt_data.iloc[idx]
        img_data = self.img_data[selected['dicom_id']]
        
        caption = selected['findings'] if self.use_captions == 'findings' else selected['impression']
        
        # label = self.label_data[self.label_data['study_id']==selected['study_id']]['labels'].values
        
        # TODO: if adding view to the input data,  preprocess to remove NaNs,
        #       else collate_fn bugs out         
        sample = {'txt': {'raw' : caption}, #'view': selected['view']},   Need to convert NaN to str to use this, or collate_fn bugs out.
                          
                  'img': {'id' : selected['dicom_id'], 
                          'features' : img_data['features'], 
                          'boxes' : img_data['boxes'],
                          'num_boxes' : img_data['num_boxes'], 
                          'img_h' : img_data['img_h'],
                          'img_w' : img_data['img_w']
                          },
                  'label': self.labels_data[idx]
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
        if topk != 0:
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

      
# import torch
# import re
# import collections
# from torch._six import string_classes

# np_str_obj_array_pattern = re.compile(r'[SaUO]')
# default_collate_err_msg_format = (
#     "default_collate: batch must contain tensors, numpy arrays, numbers, "
#     "dicts or lists; found {}")

# def debug_collate(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""

#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(default_collate_err_msg_format.format(elem.dtype))

#             return debug_collate([torch.as_tensor(b) for b in batch])
#         elif elem.shape == ():  # scalars
#             return torch.as_tensor(batch)
#     elif isinstance(elem, float):
#         try:
#             return torch.tensor(batch, dtype=torch.float64)
#         except Exception as e:
#             print("error")
#             print(e)
#             return torch.tensor(batch, dtype=torch.float64)
#     elif isinstance(elem, int):
#         return torch.tensor(batch)
#     elif isinstance(elem, string_classes):
#         return batch
    
#     elif isinstance(elem, collections.abc.Mapping):
#         return {key: debug_collate([d[key] for d in batch]) for key in elem}

#     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
#         return elem_type(*(debug_collate(samples) for samples in zip(*batch)))
#     elif isinstance(elem, collections.abc.Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError('each element in list of batch should be of equal size')
#         transposed = zip(*batch)
#         return [debug_collate(samples) for samples in transposed]

#     raise TypeError(default_collate_err_msg_format.format(elem_type))