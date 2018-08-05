from src.models.atlstm import ATLSTM
from src.utils import Logger, __fn__, load_corpus, get_envar, read_config, get_timestamp
from src.data import AbsaDataManager, LexiconManager
import numpy as np




base_configs = read_config(get_envar('CONFIG_PATH') + '/' + get_envar('BASE_CONFIG'), obj_view=True)
exp_num = 'exp_' + '4'
exp_configs = read_config(base_configs.exp_configs.path, obj_view=False)[exp_num]
hyparams = exp_configs['hyperparams']
description = exp_configs['description']
wdir = base_configs.model.path + get_timestamp() + '/'


lm = LexiconManager()
dm = AbsaDataManager(lexicon_manager=lm)

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')

hyparams['epochs'] = 2

model = ATLSTM(datamanager=dm, parameters=hyparams)


model.train(train_df, test_df)

model.predict(dev_df)


model.save(wdir)




