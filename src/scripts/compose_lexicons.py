import pandas as pd
import click
from src.utils import Logger, __fn__, list_files
from functools import reduce

logger = Logger(__fn__())

@click.command()
@click.option('--lx_dir', default='data/processed/lexicon', type=click.Path(), help='Folder contains all processed lexicons.')
@click.option('--outfile', default='lexicon_table', type=str, help='Output file name.')
def compose_lexicons(lx_dir, outfile):
    lx_files = list_files(lx_dir)
    extensions = set([f.split('.')[-1] for f in lx_files])
    assert len(extensions) ==1, 'Invalid files format {}, only accept .csv'.format(extensions)

    read_files = [f for f in lx_files if f.strip('.csv') != outfile]
    path = lx_dir + '/' if lx_dir[-1] not in {'/','\\'} else lx_dir

    logger.info('Reading lexicons: {}'.format(read_files))
    dframes = [pd.read_csv(path + f, usecols=range(2)) for f in read_files]
    logger.info('Merging lexicons with shapes: {}'.format([df.shape for df in dframes]))
    df = reduce(lambda x,y: pd.merge(x, y, how='outer', on='WORD'), dframes)
    logger.info('Writting merged lexicon to {}, merged shape: {}'.format(path, df.shape))
    df.to_csv(path + outfile + '.csv', index=False)
    logger.info('Merged lexicon saved to {}'.format(path + outfile + '.csv'))


if __name__ == '__main__':
    compose_lexicons()