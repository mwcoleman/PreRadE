import random
import torch
from collections import Counter

class PretextProcessor:
    """
    Class to perform both pre-processing (tokenisation, generate att masks) and
    pre-text manipulation of a batch of img+text (e.g. masking, retrieval...)
    """
    
    def __init__(self, tokenizer, max_seq_len=20, 
                 mlm_rate=0.15, emlm_rate=0.40,
                 mfr_rate=0.15,
                 itm_rate=0.5):

        self.mlm_rate = mlm_rate
        self.mfr_rate = mfr_rate
        self.itm_rate = itm_rate
        self.emlm_rate = emlm_rate

        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def img_vectorize(self, batch, model):
        """Creates necessary visual inputs for vbert model

        Args:
            model: PL module w/ linear projection layers (visual input processing)

        Returns:
            batch w/ projected visual inputs.
        """
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

    def tokenize_pad_vectorize(self, batch, task=None):
       
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
        batch['txt']['type_ids'] = torch.zeros_like(batch['txt']['input_ids'], device=self.device)
        
        batch['txt']['pos_ids'] = torch.ones_like(batch['txt']['input_ids'], device=self.device)
        batch['txt']['pos_ids'] *= torch.arange(0,batch['txt']['input_ids'].size()[1], 1, device=self.device)

        if task=='emlm':
            # Store the index of tokens in each sequence
            batch['txt']['word_ids'] = torch.vstack([e.word_ids for e in encoded._encodings], device=self.device)

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
                if token == "[PAD]" or token == "[SEP]":
                    sent_len = i
                    break
                if token == "[CLS]":
                    continue
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])       
            # Remaining code is as per whole word masking
            random.shuffle(cand_indexes)
            num_to_predict = min(self.max_seq_len, max(1, int(round(sent_len * self.emlm_rate))))
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
    
    
    def oov_masking(self, batch):
        """A quick and dirty approach to entity masking, assuming that 
        a general domain pretrained tokenizer (e.g. bert-base) will not recognise medical entities
        - Filter masking candidates to only those that are broken into subword tokens
        - Mask the entity (all subword tokens) up till the masking budget (oovmlm_rate) 
          is allocated on a per-sample bass
        Requires use of the Fast tokenizer class from HF. e.g. BertTokenizerFast

        Args:
            batch ([type]): The (tokenized) input batch

        Returns:
            [type]: Batch with masked_input_ids as per wwm task.
        """
        # Instantiate masked inputs
        batch['txt']['masked_input_ids'] = batch['txt']['input_ids'].detach().clone()
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['txt']['input_ids'], device=self.device)        

        # Get start and end positions of subword tokens
        def get_subwords(word_ids):
            """Returns a set of tuples of subword token position and spans
            given a sequence's corresponding word_ids
            """
            return {(word_ids.index(k),word_ids.index(k)+v-1) for 
                    k,v in Counter(word_ids[1:-1]).items() if (v>1)}
            
        subword_idxs = [get_subwords(wid) for wid in batch['txt']['word_ids']]
        

        for sample_idx,(sample_input_ids,sample_subword) in \
            enumerate(batch['txt']['input_ids'],subword_idxs):
            
            if sample_subword == set():
                # Sample has no subword tokens
                continue
            cand_indexes = [list(range(start_idx,end_idx+1)) for 
                            start_idx,end_idx in sample_subword]
            sent_len = len([t for t in sample_subword if t is not None])


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

    def span_mask(self, batch):
        """Masks contiguous spans of words up to the masking budget (mlm rate, e.g. 15%)
           Samples length of span from 

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        pass
        # # Instantiate masked inputs
        # batch['txt']['masked_input_ids'] = batch['txt']['input_ids'].detach().clone()
        # # Set targets to -100 by default to ignore
        # labels = -100 * torch.ones_like(batch['txt']['input_ids'], device=self.device)        

        # for sample_idx,sample_input_ids in enumerate(batch['txt']['input_ids']):
            
        #     # Sample lengths as following SpanBERT (Joshi et al 2020)
        #     # p: 0.2 -> mean span: 3.8
        #     span_lengths = np.random.geometric(0.2,20)
        #     span_lengths = span_lengths[span_lengths<11]

        #     input_tokens = self.tok.convert_ids_to_tokens(sample_input_ids)

        #     cand_indexes = []
        #     for (i, token) in enumerate(input_tokens):
        #         sent_len = 0
        #         if token == "[PAD]" or token == "[SEP]":
        #             sent_len = i
        #             break
        #         if token == "[CLS]":
        #             continue
        #         if len(cand_indexes) >= 1 and token.startswith("##"):
        #             cand_indexes[-1].append(i)
        #         else:
        #             cand_indexes.append([i])       
              #TODO: Up to here
        #     shuffled_idx = random.shuffle(cand_indexes)

        #     num_to_predict = min(self.max_seq_len, max(1, int(round(sent_len * self.mlm_rate))))
        #     masked_lms = []
        #     covered_indexes = set()

        #     for index_set in shuffled_idx:
        #         if len(masked_lms) >= num_to_predict:
        #             break
        #         # If adding a whole-word mask would exceed the maximum number of
        #         # predictions, then just skip this candidate.
        #         if len(masked_lms) + len(index_set) > num_to_predict:
        #             continue
        #         is_any_index_covered = False
        #         for index in index_set:
        #             if index in covered_indexes:
        #                 is_any_index_covered = True
        #                 break
        #         if is_any_index_covered:
        #             continue
        #         for index in index_set:
        #             covered_indexes.add(index)
        #             masked_lms.append(index)

        #     assert len(covered_indexes) == len(masked_lms)
        #     covered_indexes = list(covered_indexes)
        #     # mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

        #     batch['txt']['masked_input_ids'][sample_idx,covered_indexes] = torch.full((len(covered_indexes),),103, device=self.device)
        #     labels[sample_idx,covered_indexes] = sample_input_ids[covered_indexes]  
        # batch['txt']['masked_labels'] = labels.to(self.device)
        # return batch


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