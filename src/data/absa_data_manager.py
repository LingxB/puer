"""

data = DataManager(*args, **kwargs)

data.batch # batch generator to feed data into model


X, asp, lex, y



# pd.options.display.max_colwidth = 80
"""


import pandas as pd
from src.utils import load_corpus, symbolize, load_symbod_dict_if_exists, get_envar, read_config, create_dump_symbol_dict



class AbsaDataManager(object):

    def __init__(self, dataset=None, x_col=None, y_col=None, asp_col=None, lexicon_manager=None):
        self.df = dataset
        self.x_col = x_col
        self.y_col = y_col
        self.asp_col = asp_col
        self.lm = lexicon_manager
        self.config_path = get_envar('CONFIG_PATH')
        self.configs = read_config(self.config_path+'/'+get_envar('BASE_CONFIG'), obj_view=True)
        self.__initialize()


    def __initialize(self):
        self.sd = load_symbod_dict_if_exists(self.config_path, self.configs.symbol_dict.file_name)
        if self.sd == False:
            self.corpus = pd.concat([self.df[self.x_col], pd.Series(self.df[self.asp_col].unique())],
                                    ignore_index=True
                                    ).str.split()
            self.sd = create_dump_symbol_dict(self.corpus, self.configs.symbol_dict.start_idx, self.config_path, self.configs.symbol_dict.file_name)


    def token2symbol(self, sents):
        """

        Parameters
        ----------
        sents : ``list`` or ``pandas.Series`` or ``numpy.ndarray``
            An iterable (corpus) contains sentences and tokens. i.e. [['hello','world'], ['w3','w4'...], ['w8','w9',...], ...]
        Returns
        -------
        ``numpy.ndarray``
            Symbollized sentences padded with max length
        """


    def mask_unk(self):
        pass


    def bucketing(self):
        pass


    def padding(self):
        pass


    def batch_generator(self, df, batch_size):
        # TODO:
        # 1. random shuffle df
        # 2. slice into batches
        # 3. iterate over batches
        pass




df = load_corpus(['data/processed/ATAE-LSTM/train.csv',
                     'data/processed/ATAE-LSTM/dev.csv',
                     'data/processed/ATAE-LSTM/test.csv'])

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')



dm = AbsaDataManager(dataset=df, x_col='SENT', y_col='CLS', asp_col='ASP')



















df = load_corpus(['data/processed/ATAE-LSTM/train.csv',
                     'data/processed/ATAE-LSTM/dev.csv',
                     'data/processed/ATAE-LSTM/test.csv'])






d, c = create_symbol_dict(df.SENT.str.split())

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')

train = symbolize(train_df.SENT.str.split(), d)
dev = symbolize(dev_df.SENT.str.split(), d)
test = symbolize(test_df.SENT.str.split(), d)

symbolize(train_df.ASP.str.split(), d)

