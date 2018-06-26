"""
Create shrinked sym_dict and glove vector based on datasets.

01 Load files and concat into dataframe
02 Merge all columns as 'tokens'
03 Get unique words as vocab
04 Loop raw glove file, take if in vocab
05 Save shrieked sym_dict and glove_lookup table

"""


import pandas as pd
from src.utils.file_utils import save_yaml, load_corpus
from src.utils.data_utils import create_symbol_dict
import click
from src.utils import Logger, __fn__
pd.options.display.max_colwidth = 80

logger = Logger(__fn__())

@click.command()
@click.argument('files', nargs=-1, type=click.Path())
@click.argument('odir', nargs=1, type=click.Path())
@click.option('--raw_glove', default='data/raw/glove.840B.300d.txt', type=click.Path(), help='Raw glove vector file.')
@click.option('--start_idx', default=2, type=int, help='Symbol start index, default 2, 0 for padding, 1 for <UNK>')
def filter_symbol_and_glove(files, odir, raw_glove, start_idx):
    logger.info('Loading files: {}'.format(files))
    df = load_corpus(list(files)).astype(str)

    all_words = df[df.columns].apply(lambda x: ' '.join(x), axis=1)
    logger.info('Merging dataframe columns:\n{}'.format(all_words.head()))

    _sym_dict, _ = create_symbol_dict(all_words.str.split())
    vocab = _sym_dict.keys()

    data = {}
    with open(raw_glove, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
                _line = line.strip('\n').split()
                values = _line[-300:]
                key = _line[:-300]
                if len(key) > 1:
                    key = ' '.join(key)
                try:
                    key = key[0]
                except IndexError:
                    logger.debug('Empty key {} at line {}: {}'.format(key, idx+1, line[:100]+'...'))
                    key = ' '
                if key in vocab:
                    data.update({key: values})

    _df = pd.DataFrame(data).transpose().astype(float)
    _df.columns = _df.columns.astype(str)
    logger.info('Dataset vocab size: {} (including labels -1,0,1)'.format(len(vocab)))
    logger.info('Filtered embedding table ({}):\n{}'.format(_df.shape, _df.head()))

    glove_out = odir+'/glove_lookup.parquet'
    sym_out = odir+'/glove_symdict.yml'

    _df.to_parquet(glove_out)
    logger.info('Filtered glove vectors saved to {}'.format(glove_out))

    sym_dict, _ = create_symbol_dict(_df.index.str.split(), start_idx=start_idx)
    save_yaml(sym_dict, sym_out)
    logger.info('Filtered sym_dict saved to {}'.format(sym_out))




if __name__ == '__main__':
    filter_symbol_and_glove()
