from src.models.atlstm import ATLSTM
from src.utils import Logger, __fn__, load_corpus, matmul_2_3, seq_length, get_envar, read_config, get_timestamp, mkdir
from src.data import AbsaDataManager, LexiconManager
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from copy import deepcopy
from time import time
import pandas as pd



base_configs = read_config(get_envar('CONFIG_PATH') + '/' + get_envar('BASE_CONFIG'), obj_view=True)
exp_num = 'exp_' + '4'
exp_configs = read_config(base_configs.exp_configs.path, obj_view=False)[exp_num]
hyparams = exp_configs['hyperparams']
description = exp_configs['description']

lm = LexiconManager()
dm = AbsaDataManager(lexicon_manager=lm)

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')



model = ATLSTM(datamanager=dm, parameters=hyparams)


model.train(train_df, test_df)