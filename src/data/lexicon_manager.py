from src.utils import get_envar, read_config, Logger, __fn__
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


logger = Logger(__fn__())



class LexiconManager(object):


    def __init__(self, lx_path=None, usecol=-1, lx_size=-1, append_neg=False, asp_lx=False):
        """

        Parameters
        ----------
        lx_path : str
            path to lexicon table without '.csv' extension
        usecol : int () or list (list of column names)
            which lexicons to use
        """
        self.usecol = usecol
        self.lx_size = lx_size
        self.append_neg = append_neg
        self.asp_lx = asp_lx
        if lx_path is None:
            configs = read_config(get_envar('CONFIG_PATH')+'/'+get_envar('BASE_CONFIG'), obj_view=True)
            self.lx_path = configs.lexicon_table.path + '.csv'
            self.usecol = configs.lexicon_table.usecol
            self.lx_size = configs.lexicon_table.lx_size
            self.append_neg = configs.lexicon_table.append_neg
            self.asp_lx = configs.lexicon_table.asp_lx
        else:
            self.lx_path = lx_path + '.csv'
        self.__initialize()


    def __initialize(self):
        logger.info('Loading lexicon table from {}'.format(self.lx_path))
        self.lx = pd.read_csv(self.lx_path)
        assert not self.lx.duplicated().any(), 'Lexicon table has duplicated rows.'
        self.lx = self.lx.set_index(['WORD', 'ASP']) if self.asp_lx else self.lx.set_index('WORD')

        if self.usecol == -1:
            pass
        else:
            if isinstance(self.usecol, list):
                self.lx = self.lx[self.usecol]
            elif isinstance(self.usecol, int):
                self.lx = self.lx.iloc[:, :self.usecol]
            else:
                raise AttributeError('Invalid attribute usecol={}'.format(self.usecol))

        if self.lx_size == -1:
            pass
        else:
            self.lx = self.lx.iloc[:self.lx_size, :]
            logger.info('Using lexicon subset, size: {}'.format(self.lx.shape))
            if 'not' not in self.lx.index and self.append_neg:
                row = pd.Series({'MPQA': -1.0, 'OPENER': -1.0, 'OL': -1.0}, name='not')
                self.lx = pd.concat([self.lx, row.to_frame().transpose()])
                logger.info("Add 'not' to lexicon")
            if "n't" not in self.lx.index and self.append_neg:
                row = pd.Series({'MPQA': -1.0, 'OPENER': -1.0, 'OL': -1.0}, name="n't")
                self.lx = pd.concat([self.lx, row.to_frame().transpose()])
                logger.info("Add 'n't' to lexicon")

        logger.info('Using lexicon: \n{}'.format(self.lx.tail()))
        logger.info('Lexicon size: {}'.format(self.lx.shape))

    def pad_transform(self, sents: list, asps: list = None):
        if asps is None:
            return pad_sequences([self.transform(s) for s in sents], padding='post')
        else:
            assert len(sents) == len(asps)
            return pad_sequences([self.transform(s, a) for s,a in zip(sents, asps)], padding='post')


    def transform(self, sent: list, a: str = None):
        if a is None:
            return np.array(list(map(self.loc_word_pol, sent)))
        else:
            a = [a] * len(sent)
            return np.array(list(map(self.loc_word_pol, zip(sent, a))))


    def loc_word_pol(self, w, a=None):
        try:
            if a is None:
                wp = self.lx.loc[w, :].values
            else:
                wp = self.lx.loc[(w,a), :].values
        except KeyError:
            wp = np.empty(self.lx.shape[1])
            wp[:] = np.nan
        return np.nan_to_num(wp)


# lm = LexiconManager()
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