#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:05:12 2022

@author: Bryan
"""

import transformers
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm

import numpy as np
import pandas as pd
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from collections import defaultdict
# from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random as rn

# from nltk.tokenize import word_tokenize
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# import re

np.random.seed(17)
rn.seed(12345)