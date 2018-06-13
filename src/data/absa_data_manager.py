"""

data = DataManager(*args, **kwargs)

data.batch # batch generator to feed data into model

X, asp, lex, y

# pd.options.display.max_colwidth = 80
"""


import pandas as pd
import numpy as np
from src.utils import load_corpus, symbolize, load_symbod_dict_if_exists, get_envar, read_config, create_dump_symbol_dict
from src.utils import Logger, __fn__
pd.options.display.max_colwidth = 80
logger = Logger(__fn__())

class AbsaDataManager(object):

    def __init__(self, dataset=None, lexicon_manager=None, x_col='SENT', y_col='CLS', asp_col='ASP'):
        self.df = dataset
        self.x_col = x_col
        self.y_col = y_col
        self.asp_col = asp_col
        self.lm = lexicon_manager
        self.config_path = get_envar('CONFIG_PATH')
        self.configs = read_config(self.config_path+'/'+get_envar('BASE_CONFIG'), obj_view=True)
        self.__initialize()


    def __initialize(self):
        sd_path = self.configs.symbol_dict.file_path
        start_idx = self.configs.symbol_dict.start_idx
        logger.info('Loading symbol_dict from {}'.format(sd_path))
        self.sd = load_symbod_dict_if_exists(sd_path)
        if self.sd == False:
            logger.info('symbol_dict not found, creating new on: {}, {}, start_idx: {}'.format(self.x_col, self.asp_col, start_idx))
            self.corpus = pd.concat([self.df[self.x_col], pd.Series(self.df[self.asp_col].unique())], ignore_index=True).str.split()
            logger.info('dataset shape: {}'.format(self.corpus.shape))
            self.sd = create_dump_symbol_dict(self.corpus, start_idx, sd_path)
            logger.info('symbod_dict saved to {}'.format(sd_path))

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
        """

        Parameters
        ----------
        labels : iterable

        Returns
        -------
        One hot coded label vector/matrix
        """
        _labels = labels
        if not isinstance(labels, pd.Series):
            _labels = pd.Series(labels)

        return np.array(_labels.apply(self.__2dummy).tolist())

    @staticmethod
    def __2dummy(x):
        a = np.zeros(3)
        if x == -1:
            a[0] = 1
        elif x == 0:
            a[1] = 1
        elif x == 1:
            a[2] = 1
        else:
            raise IndexError('Label must be {-1, 0, 1}')
        return a

    def lex2symbol(self, sents):
        pass

    def batch_symbolizer(self, batch_df):

        _X = self.token2symbol(batch_df[self.x_col].str.split())
        _y = self.class2symbol(batch_df[self.y_col])

        if self.asp_col is not None:
            _a = self.token2symbol(batch_df[self.asp_col].str.split())
        else:
            _a = None

        if self.lm is not None:
            # TODO: add lexicon inputs
            _lx = None
        else:
            _lx = None

        return _X, _a, _lx, _y


    def batch_generator(self, df, batch_size, shuffle=False, random_state=1):
        _df = df.copy()

        if shuffle:
            _df = _df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            logger.info('Shuffled dataframe:\n{}'.format(_df.head()))

        n_batches = int(np.ceil(_df.shape[0]/batch_size))

        first_idx = np.arange(batch_size)
        last_batch = False
        for n in range(n_batches):
            if n == n_batches-1:
                last_batch = True

            batch_df, first_idx = self.__loc_update_index(_df, first_idx, last_batch)

            yield batch_df, self.batch_symbolizer(batch_df)

    @staticmethod
    def __loc_update_index(df, index, last_batch=False):
        if last_batch:
            _df = df.loc[index[0]:]
        else:
            _df = df.loc[index]
        _index = index + index.shape[0]
        return _df, _index











# df = load_corpus(['data/processed/ATAE-LSTM/train.csv',
#                      'data/processed/ATAE-LSTM/dev.csv',
#                      'data/processed/ATAE-LSTM/test.csv'])

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')

train_df = train_df.head(50)


dm = AbsaDataManager()


gen = dm.batch_generator(train_df, 32, shuffle=True)

_df, _sym = next(gen)


_X = dm.token2symbol(_df.SENT.str.split())
_a = dm.token2symbol(_df.ASP.str.split())
_y = dm.class2symbol(_df.CLS.tolist()[0])










batch1_y = train_df.CLS[:5]





batch1_x = train_df.SENT[:5].str.split()
batch2_x = train_df.SENT[5:10].str.split()

dm.token2symbol(batch1_x)

dm.token2symbol(batch2_x)


batch1_a = train_df.ASP[:5].str.split()
dm.token2symbol(batch1_a)



