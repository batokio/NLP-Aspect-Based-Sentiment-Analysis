#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:56:40 2022

@author: Bryan
"""

from imports import *

# Compute the max. sequence length to take into account in the encoding
def sequence_length(file, tokenizer):
    token_lengths = []
    for sentence in file[4]:
        tokens = tokenizer.encode(sentence, max_length=1000)
        token_lengths.append(len(tokens))
        
    # Add 30 to the max in case there are longer sequences in the dev or test files
    return max(token_lengths) + 30 

# Class to create a PyTorch dataset from the files
class create_dataset(Dataset):
    
    def __init__(self, sentences, aspect_categories, target_terms, labels, tokenizer, max_len_token):
        
        self.sentences = sentences
        self.aspect_categories = aspect_categories
        self.target_terms = target_terms
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len_token = max_len_token
        
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        aspect_category = str(self.aspect_categories[index])
        target_term = str(self.target_terms[index])
        label = self.labels[index]

        aspect = aspect_category + ' ' + target_term
        
        # encode_plus() method used to get the attention maks
        encoding = self.tokenizer.encode_plus(
            sentence,
            aspect,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=self.max_len_token,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors,
            truncation=True,
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

# Function to create a PyTorch dataloader
# The function that will leverage the class created above
def create_dataloader(df, tokenizer, max_len_token, batch_size, shuffle=True):
    
    dataset = create_dataset(
        sentences=df[4].to_numpy(),
        aspect_categories = df[1].to_numpy(),
        target_terms=df[2].to_numpy(),
        labels = df[0].to_numpy(),
        tokenizer = tokenizer,
        max_len_token = max_len_token
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=shuffle
    )
    
    return dataloader

        
