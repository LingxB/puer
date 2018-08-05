from src.models.atlstm import ATLSTM
from src.utils import Logger, __fn__, load_corpus, get_envar, read_config, get_timestamp, pickle_load
from src.data import AbsaDataManager, LexiconManager
import numpy as np


model_path = 'models/at-lstm/20180805_172714'

model = ATLSTM()

model.load(model_path)



train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')



model.predict(test_df)