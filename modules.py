import random, os, json, csv, base64, time
import torch
from torch import nn
import collections
import numpy as np
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.optim as optim

from transformers import BertTokenizer, VisualBertModel, VisualBertConfig
from transformers.models.visual_bert.modeling_visual_bert import VisualBertLMPredictionHead
class mmRadForPretraining(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.config = VisualBertConfig()
        self.model = VisualBertModel(self.config)
        self.tokenizer = BertTokenizer

        self.__init_pretraining_heads()
        self.__init_tokenizer()

    def __init_pretraining_heads(self):
        self.text_prediction_head = VisualBertLMPredictionHead
        self.seq_relationship = nn.Linear(self.config.hidden_size, 2)
        # self.image_mfr_prediction
        
    def __init_tokenizer(self, tok='bert-base-uncased'):
        
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
        
        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(loss)

        container = {'train_loss': loss, 'train_acc': acc}

        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result
    
    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on = loss)

        container = {'val_loss': loss, 'val_acc': acc}        
        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result

    def shared_step(self, batch, batch_idx):

        # a batch should be a dict containing
        # input_ids, attention_mask, token_type_ids (for seq_prediction task), position_ids,
        # visual_embeds, visual_attention_mask, return_dict=True
    
        # Need to expand batch into inputs here
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Run through PT heads - MLM only for now.
        # Most code borrowed from HF visualbertforpretraining
        sequence_output, pooled_output = outputs[:2]
        text_prediction_scores = self.text_prediction_head(sequence_output)
        # pooled_output = self.seq_relationship(pooled_output)
        total_loss = None
        
        if labels is not None:
            total_size = attention_mask.size(-1) + visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(
                    f"The labels provided should have same sequence length as total attention mask. "
                    f"Found labels with sequence length {labels.size(-1)}, expected {total_size}."
                )

            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(text_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return total_loss

class PretextProcessing:
    """Contains methods to mask etc"""
    def __init__(self, tokenizer, max_seq_len=20, mlm_rate=0.15):
        self.mlm_rate = mlm_rate
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        
    
    def tokenize_pad_vectorize(self, batch):
        batch['text_tokens'] = [self.tok.convert_tokens_to_ids(['[CLS]']+self.tok.tokenize(s.strip())+['[SEP]'])
                                for s in batch['text']]
        pad_fct = lambda a,i: a[0:i] if len(a) > i else a + [0]*(i-len(a))

        batch['input_ids'] = torch.tensor([pad_fct(sent,self.max_seq_len) for sent in batch['text_tokens']],
                                            dtype=torch.int)  
        
        return batch
    
    def mask_txt(self, batch):
        """Returns masked inputs and labels over text inputs
        Generally follows https://keras.io/examples/nlp/masked_language_modeling/"""
        inp_mask = torch.rand_like(batch['input_ids']) < self.mlm_rate

        # Avoid masking CLS(101), SEP(102) and padded (0)
        avoid_mask = torch.logical_or(batch['input_ids']==0, 
                                  batch['input_ids']==self.tok.convert_tokens_to_ids(['[CLS]']),
                                  batch['input_ids']==self.tok.convert_tokens_to_ids(['[SEP]']))
        inp_mask[avoid_mask] = False
        
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['input_ids'])
        labels[inp_mask] = batch['input_ids'][inp_mask]
        
        # Mask inputs
        masked_inputs = batch['input_ids'].copy()        
        # '[MASK]' is 103
        # of .15, 0.1 remain unchainged
        inp_mask_2m = inp_mask & (torch.rand_like(batch['input_ids']) < 0.9)

        masked_inputs[inp_mask_2m] = self.tok.convert_tokens_to_ids('[MASK]')

        # and 0.1 to random from the batch
        inp_mask_2r = inp_mask_2m & (torch.rand_like(batch['input_ids']) < 1/9)
        masked_inputs[inp_mask_2r] = np.random.choice(batch['input_ids'][~avoid_mask])

        labels = 



class CocoDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        # Add args
        self.batch_size=2

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
            self.train_tset, batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=lambda x: x,
            drop_last=self.drop_last, pin_memory=True
            # num_workers=self.num_workers
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_tset, batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            drop_last=False, pin_memory=True,
            num_workers=self.num_workers
        )
        return dl    

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
    def __init__(self, json_fp, img_ft_path):
        super().__init__()

        with open(json_fp) as f:
            self.metadata = json.load(f)
        # Dict of image fts' by id
        self.img_data = load_tsv(img_ft_path)
        # Add captions as duplicate tuples
        self.txt_data = [{'img_id':item['image_id'], 'caption':item['caption']} for item in self.metadata['annotations']]

        # self.id2captions = {item['image_id']:item['caption'] for item in self.metadata['annotations']}

    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Produces a sample per txt sequence- image features are duplicated for each.        
        img_id = str(self.txt_data[idx]['img_id'])
        caption = self.txt_data[idx]['caption']
        img_ft = self.img_data[img_id]
        sample = {'img_id': img_id, 'img_data': img_ft, 'text': caption}
        return sample



BATCH_SIZE=2
if __name__=='__main__':
    dataset = CocoDataset('/media/matt/data21/datasets/ms-coco/2017/val2017/captions_val2017.json',
                      '/media/matt/data21/mmRad/img_features/mscoco-val_2017-custom.tsv')
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    # just for testing
    sample_processor = PretextProcessing(BertTokenizer.from_pretrained('bert-base-uncased'))

    for idx,batch in enumerate(loader):
        sample = batch
        sample = sample_processor.tokenize_pad_vectorize(sample)
        break

    print("Done")