#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:49:39 2022

@author: Bryan
"""

from imports import *

# Simple function that will convert the labels to numeric values
def preprocessing(file):
    
    # Negative == 0
    # Neutral == 1
    # Positive == 2
    
    file[0] = file[0].apply(lambda x: 0 if x == "negative" else (1 if x=="neutral" else 2))
    
    # Convert the aspect column to examples/questions
    file[1] = file[1].apply(lambda x: aspect_categories(x))
    
    
    return file


# This function will convert the aspect categories to examples that will then be used for the encoding of the each sentence
#The examples will be questions as the sentences seem to be some kind of answers
def aspect_categories(aspect):
    
  if aspect == 'AMBIENCE#GENERAL':
    return "What do you think of the ambience ?"

  elif aspect == 'FOOD#QUALITY':
    return "What do you think of the quality of the food ?"

  elif aspect == 'SERVICE#GENERAL':
    return "What do you think of the service ?"

  elif aspect == 'FOOD#STYLE_OPTIONS':
    return "What do you think of the food choices ?"

  elif aspect == 'DRINKS#QUALITY':
    return "What do you think of the drinks?"

  elif aspect == 'RESTAURANT#MISCELLANEOUS' or aspect == 'RESTAURANT#GENERAL':
    return "What do you think of the restaurant ?"

  elif aspect == 'LOCATION#GENERAL':
    return 'What do you think of the location ?'

  elif aspect == 'DRINKS#STYLE_OPTIONS':
    return "What do you think of the drink choices ?"
  
  elif aspect == 'RESTAURANT#PRICES' or aspect =='DRINKS#PRICES' or aspect == 'FOOD#PRICES':
    return 'What do you think of the price of it ?'
    
    