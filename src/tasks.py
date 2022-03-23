import random
import torch
import numpy as np
from collections import Counter

class PretextProcessor:
    """
    Class to perform both pre-processing (tokenisation, generate att masks) and
    pre-text manipulation of a batch of img+text (e.g. masking, retrieval...)
    """
    
    def __init__(self, tokenizer, max_seq_len=125, 
                 mlm_rate=0.15, oovm_rate=0.40,
                 mfr_rate=0.15, span_rate=0.15,
                 itm_rate=0.5):

        self.mlm_rate = mlm_rate
        self.mfr_rate = mfr_rate
        self.itm_rate = itm_rate
        self.oovm_rate = oovm_rate
        self.span_rate = span_rate
        # Set max targets for each span masking
        self.span_max_targets = 10


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

    def tokenize_pad_vectorize(self, batch, return_word_ids=False):
       
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

        if return_word_ids:
            # Store the index of tokens in each sequence
            batch['txt']['word_ids'] = [[-1 if x is None else x for x in w] for w in [e.word_ids for e in encoded._encodings]]

        return batch
    
    def mask_token(self, batch):
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

    def mask_whole_word(self,batch):
        """Returns masked inputs and labels over text inputs
        samples from candidate whole words not parts of.
        batch: training data
        returns: batch w/ masked whole words.
        Processes on per-example basis, may be slow.
        Roughly follows https://github.com/huggingface/transformers/blob/07708793f20ec3a949ccab32cc4fe0c7272dcc4c/src/transformers/data/data_collator.py#L301"""

        bs,seq_len = batch['txt']['input_ids'].shape        
        # Instantiate masked inputs
        batch['txt']['masked_input_ids'] = batch['txt']['input_ids'].detach().clone()
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['txt']['input_ids'], device=self.device)        
        mask_label = torch.zeros_like(labels, device='cpu')
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
            # covered_indexes = torch.tensor(list(covered_indexes), dtype=int)
            mask_label[sample_idx] = torch.tensor([1 if i in covered_indexes else 0 for i in range(len(input_tokens))])


        # batch['txt']['masked_input_ids'][sample_idx,covered_indexes] = torch.full((len(covered_indexes),),103, device=self.device)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask_label.bool()
        batch['txt']['masked_input_ids'][indices_replaced] = torch.full_like(batch['txt']['masked_input_ids'][indices_replaced],
                                                                             self.tok.convert_tokens_to_ids(self.tok.mask_token))
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & mask_label.bool() & ~indices_replaced

        r_idx = torch.randint(0,bs*seq_len, (len(indices_random[indices_random>0]),))
        batch['txt']['masked_input_ids'][indices_random] = batch['txt']['input_ids'].view(-1)[r_idx]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged    
        # --

        labels[mask_label.bool()] = batch['txt']['input_ids'][mask_label.bool()]  
        batch['txt']['masked_labels'] = labels.to(self.device)
        return batch
    
    def get_subwords(self,word_ids):
        """Returns a set of tuples of subword token start position and corresponding token span
        given a sequence's word_ids
        """
        return {(word_ids.index(k),word_ids.index(k)+v-1) for 
                k,v in Counter(word_ids[1:-1]).items() if (v>1)}


    def mask_oov_word(self, batch):
        """A quick and dirty approach to entity masking, assuming that 
        a general domain pretrained tokenizer (e.g. bert-base) will not recognise medical entities
        - Filter masking candidates to only those that are broken into subword tokens
        - Mask the entity (all subword tokens) up till the masking budget (oovmlm_rate) 
          is allocated on a per-sample bass
        Requires use of the Fast tokenizer class from HF. e.g. BertTokenizerFast

        Args:
            batch (dict): The (tokenized) input batch

        Returns:
            batch (dict): Batch with masked_input_ids as per wwm task.
        """
        # Instantiate masked inputs
        batch['txt']['masked_input_ids'] = batch['txt']['input_ids'].detach().clone()
        # Set targets to -100 by default to ignore
        labels = -100 * torch.ones_like(batch['txt']['input_ids'], device=self.device)        
        # Get start and end positions of subword tokens
        subword_idxs = [self.get_subwords(wid) for wid in batch['txt']['word_ids']]

        for sample_idx,(sample_input_ids,sample_subword) in \
            enumerate(zip(batch['txt']['input_ids'],subword_idxs)):
            
            if sample_subword == set():
                # Sample has no subword tokens
                continue
            cand_indexes = [list(range(start_idx,end_idx+1)) for 
                            start_idx,end_idx in sample_subword]
            sent_len = len([t for t in sample_input_ids if (t > 0)])-2


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

    def mask_span(self, batch):
        """Masks contiguous spans of words up to the masking budget (mlm rate, e.g. 15%)
           Note: masking budget is in words, not tokens.

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """

        # SpanBERT task is P(Wi | Ws-1, We+1, Pi ) 
        # i.e. predict token Wi given the boundary words (of the span) and the position embedding

        # Set number to predict for a sentence (19)
        num_to_predict = int(round(self.max_seq_len*self.span_rate))

        # Set #spans constant within a batch.
        # p: 0.2 -> mean span: 3.8
        span_lengths = np.random.geometric(0.2,20) 
        span_lengths = span_lengths[span_lengths<11]
        span_lengths = span_lengths[span_lengths>1]
        span_lengths = [s for i,s in enumerate(span_lengths) if sum(span_lengths[:i+1])<=num_to_predict]

        num_spans = len(span_lengths)
        batch_size = batch['txt']['input_ids'].size()[0]
        
        # set targets to -100 to ignore by default
        labels = -100 * torch.ones((batch_size*num_spans,self.span_max_targets), device=self.device, dtype=int)

        # Set up pairs
        pairs = torch.zeros((batch_size, num_spans, 2), device=self.device, dtype=int)

        for sample_idx,sample_input_ids in enumerate(batch['txt']['input_ids']):


            input_tokens = self.tok.convert_ids_to_tokens(sample_input_ids)

            cand_indexes = []
            sent_len=0
            for (i, token) in enumerate(input_tokens):
                if token == "[PAD]" or token == "[SEP]":
                    sent_len=i
                    break
                if token == "[CLS]":
                    continue
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            
            if (len(cand_indexes) < num_spans*2+1):# or (sent_len<((num_spans*2)+1)):
                # |_||_||_||_|
                # Skip masking this example, it's too short
                continue

            # distributing spans evenly across sequence
            cand_split_by_spans = [list(a) for a in np.array_split(np.array(cand_indexes, dtype=object),
                                                                    len(span_lengths))]   
            
            # adjust any span lengths that > sequence split
            adj_span_lengths = [min(span,(split[-1][-1]-split[0][0])-2) for span,split in zip(span_lengths, cand_split_by_spans)]
            
            # Adjust first split start position account for CLS
            cs_expanded = [range(max(2,e[0][0]), e[-1][-1]+1) for e in cand_split_by_spans]
            
            # Choose any val in range of sp
            start_idxs = [random.choice(range(split[0],split[-1]-(sl-1))) for split,sl in zip(cs_expanded, adj_span_lengths)]

            ww_idxs = [e[0] for e in cand_indexes]
            
            lefts, rights = [], []
            for span,start in zip(adj_span_lengths,start_idxs):
                # Find nearest whole word boundary w/out exceeding span length and 
                adj_start = min([x for x in ww_idxs if (x <= start)],
                           key=lambda x: abs(start-x))
                
                adj_end = min([x for x in ww_idxs if x < adj_start+span],
                           key=lambda x: abs(adj_start+span-x))

                # start,end are inner boundaries span boundaries
                lefts.append(adj_start-1)
                rights.append(adj_end+1)
                masks = [i for i in range(start, adj_end+1)]
                adj_span = len(masks)
                # Labels need shape (bs*num_spans, seq_len)
                labels[(sample_idx*num_spans)+len(lefts)-1, 0:adj_span] = sample_input_ids[masks]

            lefts, rights = torch.tensor(lefts, device=self.device, dtype=int), torch.tensor(rights, device=self.device, dtype=int)
            

            pairs[sample_idx,:,0], pairs[sample_idx,:,1], = lefts, rights

        batch['txt']['span_pairs'] = pairs
        batch['txt']['masked_labels'] = labels.to(self.device)
        # The encoder runs on the entire input
        batch['txt']['masked_input_ids'] = batch['txt']['input_ids']
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

    ### New tasks

    def patch_corruption(self, batch):
        """Randomly replace a token (image or text) with one from another sample
        and set is_matched labels for a lower level ITM task"""
        batch_size = len(batch['img']['id'])

        sample_select_idx = torch.bernoulli(torch.full((batch_size,), 0.75)).bool()
        # 1/3 time replace text, 1/3 time replace image, 1/3 time replace both
        idx_text = torch.bernoulli(torch.full((batch_size,), 0.33)).bool() & sample_select_idx
        idx_img = torch.bernoulli(torch.full((batch_size,), 0.33)).bool() & sample_select_idx & ~idx_text
        idx_both = sample_select_idx.bool() & ~idx_text & ~idx_img
        
        idx_text = (idx_text | idx_both)
        idx_img = (idx_img | idx_both)
        
        def swap_embed(inp,swap_idxs,negative_samples):
            """ Helper to swap one element of inp (n-d tensor), taken from each row indexed by swap_idx, 
            with a random element from negative_samples (a 1d tensor). Returns modified inp 
            """
            rand_idx = torch.randint(1,len(negative_samples), (sum(swap_idxs),))
            if len(inp.shape)>2: # Image
                rand_mask = torch.randint(0,36,(sum(swap_idxs),))
            else:
                # Avoid selecting special text tokens (pad, cls, sep)
                rand_mask = [random.randint(1,len(sample[sample>0])-1) for sample in inp[swap_idxs]]
            inp[swap_idxs,rand_mask] = negative_samples[rand_idx]
            return inp      
        
        ## Txt sampling
        masked_input_ids = batch['txt']['input_ids'].detach().clone() 
        # Avoid swapping CLS(101), SEP(102) and padded (0)
        avoid_mask = torch.add(batch['txt']['input_ids']==0,
                               torch.add(batch['txt']['input_ids']==101,
                                         batch['txt']['input_ids']==102))
        batch['txt']['input_ids'] = swap_embed(batch['txt']['input_ids'], idx_text, masked_input_ids[~avoid_mask])

        ## Img sampling
        batch['img']['features'] = swap_embed(batch['img']['features'], idx_img, batch['img']['features'].view(-1,1024))
        batch['img']['boxes'] = swap_embed(batch['img']['boxes'], idx_img, batch['img']['features'].view(-1,4))
        
        # is_matched is now a 3 class: 1 (yes), 2 (txt corrupt), 3 (img corrupt), 0 (both corrupt)
        batch['is_matched'] = torch.ones((batch_size,), device=self.device)
        batch['is_matched'][idx_text] = torch.full((sum(idx_text),),2., device=self.device)
        batch['is_matched'][idx_img] = torch.full((sum(idx_img),),3., device=self.device)
        batch['is_matched'][idx_both] = torch.zeros((sum(idx_both),), device=self.device)
        # expand dim to match the seq_rel linear layer output
        batch['is_matched'] = torch.unsqueeze(batch['is_matched'],1).long()
        
        return batch
