import random, os, json, csv, base64, time
import torch
from torch import nn
import collections
import numpy as np
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.optim as optim

from transformers import BertTokenizer, VisualBertModel, VisualBertConfig, VisualBertPreTrainedModel
from transformers.models.visual_bert.modeling_visual_bert import VisualBertLMPredictionHead
# TEMP:
from transformers.models.visual_bert.modeling_visual_bert import *


class TempVisualBertModel(VisualBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = TempVisualBertEmbeddings(config)
        self.encoder = VisualBertEncoder(config)

        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image.
            >>> from transformers import BertTokenizer, VisualBertModel
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
            >>> inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
            >>> inputs.update({
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... })
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if visual_embeds is not None and visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if visual_embeds is not None:
            combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=-1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                combined_attention_mask, [batch_size, input_shape + visual_input_shape], device
            )

        else:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, [batch_size, input_shape], device
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
        )

        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, text_length, :text_length]

            encoded_outputs = self.encoder(
                text_embedding_output,
                attention_mask=text_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoded_outputs[0]
            concatenated_input = torch.cat((sequence_output, visual_embedding_output), dim=1)
            sequence_output = self.additional_layer(concatenated_input, extended_attention_mask)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]

            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class TempVisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(
                    visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device
                )

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    logger.warning(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids
                )
            else:
                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class mmRadForPretraining(pl.LightningModule):
    def __init__(self, args, tokenizer='bert-base-uncased'):
        super().__init__()

        self.args = args
        # TODO: Access from args / elsewhere:
        self.config = VisualBertConfig(visual_embedding_dim=1024)
        # Extracted features dim
        self.visual_features_dim = 1024
        self.max_seq_len = 20
        # TODO: Remove temp
        self.model = VisualBertModel(self.config)#.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        self.__init_pretraining_heads()
        self.__init_tokenizer(tok=tokenizer)
        self.__init_transforms()

        self.pp = PretextProcessor(self.tokenizer)

    def __init_pretraining_heads(self):
        self.text_prediction_head = VisualBertLMPredictionHead(self.config)
        self.seq_relationship = nn.Linear(self.config.hidden_size, 2)
        # self.image_mfr_prediction

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

    def training_step(self, batch, batch_idx):
        # Preprocess based on the pretext tasks
        loss, acc = self.shared_step(batch, batch_idx)
        # result = pl.TrainResult(loss)

        container = {'train_loss': loss, 'train_acc': acc}

        # result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        # result = pl.EvalResult(checkpoint_on = loss)

        container = {'val_loss': loss, 'val_acc': acc}        
        # result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return loss

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
        visual_attention_mask=torch.ones((len(batch), num_features)).to(self.device)
        visual_token_type_ids=torch.zeros((len(batch), num_features)).to(self.device)
       
        # just make labels = inputs (for now)
        visual_labels = visual_embeds

        # labels = torch.hstack([batch['txt']['input_ids'], visual_labels])
        
        # MLM only for now
        labels = batch['txt']['masked_labels']

        outputs = self.model(
            input_ids=batch['txt']['masked_input_ids'],
            attention_mask=batch['txt']['att_mask'],
            token_type_ids=batch['txt']['type_ids'],
            position_ids=batch['txt']['pos_ids'],
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
        text_logits = text_prediction_scores[:,:self.max_seq_len,:]
        text_preds = text_logits[(labels > 0), :].argmax(1)
        filtered_labels = labels[(labels > 0)]

        # pooled_output = self.seq_relationship(pooled_output)
        total_loss = None
        
        if labels is not None:
            # TODO: Increase size for txt+image
            total_size = batch['txt']['att_mask'].size(-1) #+ visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(
                    f"The labels provided should have same sequence length as total attention mask. "
                    f"Found labels with sequence length {labels.size(-1)}, expected {total_size}."
                )

            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(text_logits.contiguous().view(-1, self.config.vocab_size), labels.view(-1))
            acc = (text_preds == filtered_labels).type(torch.float).mean()*100
        return total_loss,acc

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)
        optimizer = optim.Adam(self.model.parameters(), lr = 2e-5)

        return [optimizer]


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
        encoded = [self.tok.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length = self.max_seq_len,
            truncation=True,
            pad_to_max_length = True,
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
                               torch.add(batch['txt']['input_ids']==self.tok.convert_tokens_to_ids(['[CLS]']),
                                         batch['txt']['input_ids']==self.tok.convert_tokens_to_ids(['[SEP]'])))
        

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

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        # Add args
        self.batch_size=2
        self.valid_batch_size=2
        self.shuffle=True
        self.drop_last = False
        self.num_workers = 12

    def prepare_data(self):
        # Called on 1 GPU only

        pass

    def setup(self, stage=None):
        # Called on every GPU
        if stage=='fit' or stage is None:
            self.train_dset = CocoDataset('/media/matt/data21/datasets/ms-coco/2017/val2017/captions_val2017.json',
                      '/media/matt/data21/mmRad/img_features/mscoco-val_2017-custom.tsv')
            self.valid_dset = CocoDataset('/media/matt/data21/datasets/ms-coco/2017/val2017/captions_val2017.json',
                      '/media/matt/data21/mmRad/img_features/mscoco-val_2017-custom.tsv')

        if stage=='test' or stage is None:
            pass

        # self.dims = len(self.train_dset)//self.batch_size

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dset, batch_size=self.batch_size,
            shuffle=self.shuffle,
            # collate_fn=lambda x: x,
            drop_last=self.drop_last, pin_memory=True
            # num_workers=self.num_workers
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_dset, batch_size=self.valid_batch_size,
            shuffle=False,
            # collate_fn=lambda x: x,
            drop_last=False, pin_memory=True,
            #num_workers=self.num_workers
        )
        return dl    


from pytorch_lightning.callbacks import Callback

class MyCallBack(Callback):
    def on_train_end(self, trainer, pl_module):
        print("\n")

def load_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A dict of image object features where each feature is a dict.
    """
    import sys
    csv.field_size_limit(sys.maxsize)
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, ["img_id", "img_h", "img_w", 
                        "num_boxes", "boxes", "features"], delimiter="\t")
        
        data = {}
        for _, item in enumerate(reader):
            new_item = {}
            num_boxes = int(item['num_boxes'])
            for key in ['img_h', 'img_w', 'num_boxes']:
                new_item[key] = int(item[key])
            # slice from 2: to remove b' (csv.writer wraps all vals in str())
            new_item['features'] = np.frombuffer(base64.b64decode(item['features'][2:]), dtype=np.float32).reshape(num_boxes,-1).copy()
            new_item['boxes'] = np.frombuffer(base64.b64decode(item['boxes'][2:]), dtype=np.float32).reshape(num_boxes,4).copy()
            
            data[item['img_id']] = new_item
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

class CocoDataset(Dataset):
    """MS-COCO dataset captions only
    No transforms/process here"""
    def __init__(self, json_fp, img_ft_path, topk=None):
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


# Sample run 
BATCH_SIZE=256
if __name__=='__main__':
    # dataset = CocoDataset('/media/matt/data21/datasets/ms-coco/2017/val2017/captions_val2017.json',
    #                   '/media/matt/data21/mmRad/img_features/mscoco-val_2017-custom.tsv')
    # loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    # # just for testing
    # sample_processor = PretextProcessor(BertTokenizer.from_pretrained('bert-base-uncased'))

    # for idx,batch in enumerate(loader):
    #     sample = batch
    #     sample = sample_processor.tokenize_pad_vectorize(sample)
    #     sample = sample_processor.mask_txt(sample)
    #     break

    # print("Done")

    #### pl:
    dm = CocoDataModule(BATCH_SIZE)
    mmRad = mmRadForPretraining(args=None)

    trainer = pl.Trainer(gpus=1, callbacks=[MyCallBack()])
    trainer.fit(mmRad, dm)
    print('hi')
