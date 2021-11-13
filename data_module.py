import random
import torch
import collections
import numpy as np
from torch.utils.data import DataLoader
# import torch.nn.functional as F
import pytorch_lightning as pl

###
# Predefined Model componenents 
# from tokenization import BertTokenizer  # Full end to end tokenization, wordpiece as per BERT
from transformers import PreTrainedModel, LxmertConfig, LxmertModel, BertTokenizer

class PreTrainDataModule(pl.LightningDataModule):

    def __init__(self,args=None):
        super().__init__()

        # TODO: arg parser

    def prepare_data(self):
        # Called on 1 gpu. No self. assignments here
        pass

    def setup(self, stage=None):

        if stage=='fit' or stage is None:
            

        if stage=='test':
            pass

