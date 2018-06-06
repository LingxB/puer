"""

data = DataManager(*args, **kwargs)

data.batch # batch generator to feed data into model

X, asp, lex, y

# pd.options.display.max_colwidth = 80
"""


import pandas as pd
from src.utils import load_corpus, symbolize, load_symbod_dict_if_exists, get_envar, read_config, create_dump_symbol_dict
from src.utils import Logger, __fn__

logger = Logger(__fn__())

class AbsaDataManager(object):

    def __init__(self, dataset=None, lexicon_manager=None, x_col='SENTS', y_col='CLS', asp_col='ASP'):
        self.df = dataset
        self.x_col = x_col
        self.y_col = y_col
        self.asp_col = asp_col
        self.lm = lexicon_manager
        self.config_path = get_envar('CONFIG_PATH')
        self.configs = read_config(self.config_path+'/'+get_envar('BASE_CONFIG'), obj_view=True)
        self.__initialize()


    def __initialize(self):
        logger.info('Loadding symbol_dict from {}'.format(self.configs.symbol_dict.file_path))
        self.sd = load_symbod_dict_if_exists(self.configs.symbol_dict.file_path)
        if self.sd == False:
            logger.info('symbol_dict not found, creating new on: {}, {}, start_idx: {}'.format(self.x_col, self.asp_col, self.configs.symbol_dict.start_idx))
            self.corpus = pd.concat([self.df[self.x_col], pd.Series(self.df[self.asp_col].unique())],
                                    ignore_index=True
                                    ).str.split()
            self.sd = create_dump_symbol_dict(self.corpus, self.configs.symbol_dict.start_idx, self.configs.symbol_dict.file_path)
            logger.info('symbod_dict saved to {}'.format(self.configs.symbol_dict.file_path))

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
        return symbolize(sents, self.sd)

    def class2symbol(self, labels):
        pass


    def batch_generator(self, df, batch_size):
        # TODO:
        # 1. random shuffle df
        # 2. slice into batches
        # 3. iterate over batches
        pass




# df = load_corpus(['data/processed/ATAE-LSTM/train.csv',
#                      'data/processed/ATAE-LSTM/dev.csv',
#                      'data/processed/ATAE-LSTM/test.csv'])

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')


dm = AbsaDataManager()

batch1_x = train_df.SENT[:5].str.split()
batch2_x = train_df.SENT[5:10].str.split()

dm.token2symbol(batch1_x)

dm.token2symbol(batch2_x)


batch1_a = train_df.ASP[:5].str.split()
dm.token2symbol(batch1_a)



