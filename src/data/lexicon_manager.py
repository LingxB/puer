from src.utils import get_envar, read_config, Logger, __fn__
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


logger = Logger(__fn__())



class LexiconManager(object):


    def __init__(self, lx_path=None, usecol=-1):
        self.usecol = usecol
        if lx_path is None:
            configs = read_config(get_envar('CONFIG_PATH')+'/'+get_envar('BASE_CONFIG'), obj_view=True)
            self.lx_path = configs.lexicon_table.path + '.csv'
            self.usecol = configs.lexicon_table.usecol
        else:
            self.lx_path = lx_path + '.csv'
        self.__initialize()


    def __initialize(self):
        logger.info('Loading lexicon table from {}'.format(self.lx_path))
        self.lx = pd.read_csv(self.lx_path)
        assert not self.lx.duplicated().any(), 'Lexicon table has duplicated keys.'
        self.lx = self.lx.set_index('WORD')

        if self.usecol == -1:
            pass
        else:
            if isinstance(self.usecol[0], str):
                self.lx = self.lx[self.usecol]
            elif isinstance(self.usecol[0], int):
                self.lx = self.lx.iloc[:, self.usecol]
            else:
                raise AttributeError('Invalid attribute usecol={}'.format(self.usecol))
        logger.info('Using lexicon: \n{}'.format(self.lx.head()))


    def pad_transform(self, sents):
        return pad_sequences([self.transform(s) for s in sents], padding='post')


    def transform(self, sent):
        return np.array(list(map(self.loc_word_pol, sent)))


    def loc_word_pol(self, w):
        try:
            wp = self.lx.loc[w, :].values
        except KeyError:
            wp = np.empty(self.lx.shape[1])
            wp[:] = np.nan
        return np.nan_to_num(wp)


# lm = LexiconManager()
#
#
# sents = ['hello world abnormal ! abandoned'.split(),
#          'wobble abhor wasting whore'.split()
#          ]
#
# lm.pad_transform(sents)
#
#
# lm.transform(sents[1])
#
# lm.loc_word_pol(sents[1][1])