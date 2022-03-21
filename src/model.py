#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:40:21 2022

@author: Bryan
"""

from imports import *

class SentimentClassifier(nn.Module):
    
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        PRE_TRAINED_MODEL = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
        )[0:2]
        output = self.drop(pooled_output)
        return self.out(output)