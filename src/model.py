import os, random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, GELU, BCELoss
import pytorch_lightning as pl
from ast import literal_eval

from transformers import (
    BertTokenizer, 
    VisualBertModel, 
    VisualBertConfig, 
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.models.visual_bert.modeling_visual_bert import VisualBertLMPredictionHead

from src.tasks import PretextProcessor


class MMRad(pl.LightningModule):
    """Base framework class, e.g. MMRadForPretraining and 
       MMRadForClassification inherits from this.

       Contains methods to initialise tokeniser, encoder (e.g. VisualBert from huggingface),
       relevant linear heads and visual input transforms,
       and setup the optimiser with scheduler (wadam with LR scheduler by default)
    """
  
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        """
        Args:
            args (namespace): namespace generated by argparse, see parameters.py
            train_size (int): size of the training dataset
            tokenizer (str, optional): tokenizer compatible with HF. Defaults to 'bert-base-uncased'.
        """
        super().__init__()
        self.save_hyperparameters(args)

        self.config = VisualBertConfig(
            visual_embedding_dim=self.hparams.visual_embedding_dim,
            num_attention_heads=self.hparams.num_attention_heads,
            dropout=self.hparams.dropout,
            hidden_size=self.hparams.encoder_hidden_size,
            num_hidden_layers=self.hparams.num_tx_layers
            )
        # Extracted features
        self.visual_features_dim = self.hparams.extracted_ft_dim

        self.train_size = train_size
        
        if self.hparams.load_model is None:
            print(f"Initialising Tx encoder from scratch\n")
            self.model = VisualBertModel(self.config)
            self.model.apply(self.init_weights)
        else:
            model_path = self.hparams.load_model
            print(f"Loading transformer encoder from {model_path}\n")
            self.model = VisualBertModel(self.config).from_pretrained(model_path, config=self.config)

        if self.hparams.freeze:
            print("Freezing encoder layers..")
            for param in self.model.parameters():
                param.requires_grad = False

        self._init_tokenizer(tok=tokenizer)
        self._init_transforms()
        # All pretext data aug tasks contained here
        self.pp = PretextProcessor(self.tokenizer, max_seq_len=self.hparams.max_seq_len)
        # self.config.visual_embedding_dim will be overidden by a preloaded model.
        self.config.visual_embedding_dim = self.model.embeddings.visual_projection.in_features

    def _init_transforms(self):
        """Linear transform of input embeddings (from obj. detector
        to input of Tx encoder."""
        
        self.transform_img_ft = nn.Sequential(
            nn.Linear(self.visual_features_dim, self.config.visual_embedding_dim),
            nn.LayerNorm(self.config.visual_embedding_dim)
        )
        self.transform_img_box = nn.Sequential(
            nn.Linear(4, self.config.visual_embedding_dim),
            nn.LayerNorm(self.config.visual_embedding_dim)
       )
        self.transform_img_box.apply(self.init_weights)
        self.transform_img_ft.apply(self.init_weights)
    
    def vis_pos_embeds(self, img_ft, img_box):
        """Projects and combines (avg) the image features and box positions
           to the dimension required by the encoder.
           Input features must be .tsv file generated by extract_features.py
           (uses detectron2)
        Args:
            img_ft (np.array): image features loaded from .tsv file
            img_box (np.array): box features loaded from .tsv file

        Returns:
            (np.array): combined feature vectors for input into model (no position encodings)
        """
        embed_ft = self.transform_img_ft(img_ft)
        embed_pos = self.transform_img_box(img_box)
        return torch.div(torch.add(embed_ft, embed_pos), 2)
    
    def _init_tokenizer(self, tok):
        """Load the tokenizer

        Args:
            tok (str): either path to local tokeniser or 
            HF compatible model e.g. 'bert-base-uncased'
        """
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
        """Initialise the weights as per BERT

        Args:
            module (nn.Module): module to initialise
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, **inputs):
        return self.model(**inputs)

   
    def configure_optimizers(self):
        """Set up the optimiser and LR scheduler
           by default AdamW and linear scheduler with warmup 0.15

        Returns:
            (optimizer, scheduler)
        """
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
    """PL class for pretraining multimodal transformer (single stream) such as VisualBert.
       Inherits from MMRad
       Contains methods to make predictions on specified SSL pretext tasks (e.g. MLM, MFR)
       and logs results to logger (wandb)
    """
    def __init__(self, args, train_size, tokenizer='bert-base-uncased'):
        super().__init__(args, train_size, tokenizer=tokenizer)

        self.__init_pretraining_heads()
        self.task_step = {'mlm':self.mlm_step, 'mfr':self.mfr_step, 'itm':self.itm_step,
                          'wwm':self.wwm_step}
        self.hparams.tasks = literal_eval(self.hparams.tasks)


    def __init_pretraining_heads(self):
        """Initialise the task-specfic heads required for pretraining (e.g. MLM)
        """
        self.text_prediction_head = VisualBertLMPredictionHead(self.config)
        self.seq_relationship_head = nn.Linear(self.config.hidden_size, 2)
        self.image_mfr_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.visual_features_dim),
                GELU(),
                nn.LayerNorm(self.visual_features_dim, eps=1e-12)
                )
        
        for head in (self.text_prediction_head, 
                     self.seq_relationship_head, 
                     self.image_mfr_head):

            head.apply(self.init_weights)


    def wwm_step(self, batch, batch_idx):
        return self.mlm_step(batch, batch_idx, subtask='wwm')

    def mlm_step(self, batch, batch_idx, subtask='mlm'):
        """Computes forward pass and loss for all text related prediction tasks
           (MLM, WWM, Span-MLM, ...)

        Args:
            batch (dict)
            batch_idx (int)
            subtask (str, optional): the type of text-masking task. Defaults to 'mlm'.

        Returns:
            (dict): Dictionary containing loss and accuracy for the batch.
        """
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
        """Generic training step on a batch. 

           For each batch, samples a task from hparams.tasks (list) 
           with uniform probability, and calls the relevant forward step. 
           (Not a composite objective function each batch)
           
           Logs the resulting loss and metric
        
        Returns:
            (float): the loss
        """
        # Sample a pretext task for each iteration
        task = random.choice(self.hparams.tasks)
        batch = self.pp.tokenize_pad_vectorize(batch)
        # Preprocess based on the pretext tasks
        metrics = self.task_step[task](batch, batch_idx) #self.shared_step(batch, batch_idx, task)
        logs = {'train_'+task+'_'+k:v for k,v in metrics.items()}
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True,
                      logger = True, batch_size = self.hparams.batch_size)
        return metrics['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step

        unlike training step, validation step performs forward pass on all tasks

        Returns:
            (float): The average loss across all tasks.
        """
        tot_loss = 0
        logs = {}

        batch = self.pp.tokenize_pad_vectorize(batch)
        # Validation step calculates loss for all selected tasks
        for task in self.hparams.tasks:
            metrics = self.task_step[task](batch, batch_idx) #self.shared_step(batch, batch_idx, task)
            tot_loss += metrics['loss']

            task_log = {'val_'+task+'_'+k:v for k,v in metrics.items()}
            logs.update(task_log)
            
        logs['val_avg_loss'] = tot_loss/len(self.hparams.tasks)
        self.log_dict(logs, on_step = True, on_epoch = True, prog_bar = True, 
                      logger = True, batch_size = self.hparams.valid_batch_size)
        return logs['val_avg_loss']


class MMRadForClassification(MMRad):
    """PL module for (supervised) classification fine tuning / eval
    Inherits from mmRad
    
    - Adds head for image classification

    Args:
        MMRad ([type]): [description]
    """
    def __init__(self, args, train_size, n_classes, labelset=None, n_hidden=512, 
                 tokenizer='bert-base-uncased'):
        """
        Args:
            args (namespace): see parameters.py
            train_size (int): size of training dataset
            n_classes (int): Number of classes/labels (multilabel)
            labelset (list, optional): ordered list of the label names. Defaults to None.
            n_hidden (int, optional): dim of encoder output. Defaults to 512.
            tokenizer (str, optional): tokenizer. Defaults to 'bert-base-uncased'.
        """
        super().__init__(args, train_size, tokenizer=tokenizer)
        
        self.labelset = labelset
        
        # self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.cls = nn.Linear(self.config.hidden_size, n_classes)
        # self.cls.apply(self.init_weights)

        self.cls = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            GELU(),
            nn.LayerNorm(self.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.config.hidden_size * 2, n_classes)
        )
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

    
    def validation_step(self, batch, batch_idx):

        metrics = self.shared_step(batch, batch_idx)
      
        # Log per-step metrics
        logs = {'val_'+k:v for k,v in metrics.items() if k!='preds'}
        self.log_dict(logs, on_step = False, on_epoch = True, 
                      prog_bar = True, logger = True, batch_size = self.hparams.valid_batch_size)
        return {'loss': metrics['loss'], 'preds':metrics['preds']}

    def test_step(self, batch, batch_idx):

        metrics = self.shared_step(batch, batch_idx)
      
        # Log per-step metrics
        logs = {'test_'+k:v for k,v in metrics.items() if k!='preds'}
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
        
        num_features = batch['img']['num_boxes'][0]
        ## img input to tx dim and add positions
        visual_embeds = self.vis_pos_embeds(img_ft=batch['img']['features'],
                                            img_box=batch['img']['boxes'])
        
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

        if self.hparams.txt_only:
            visual_attention_mask=torch.zeros((len(batch['img']['id']), num_features), device=self.device)
        else:
            visual_attention_mask=torch.ones((len(batch['img']['id']), num_features), device=self.device)

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
          
        logits = self.cls(pooled_output)
        preds = nn.Sigmoid()(logits) 
        
        loss_fct = nn.BCEWithLogitsLoss()
        if self.hparams.easy_classification:
            logits = logits.view(-1)
            loss = loss_fct(logits, labels)
            acc = ((preds.view(-1) > 0.5) == labels).type(torch.float).mean()*100
            
        else:
            loss = loss_fct(logits, labels)
            acc = ((preds > 0.5) == labels).type(torch.float).mean(dim=0)

        metrics = {'loss':loss, 'acc':acc, 'preds':preds}
        return metrics