from src.utils import Logger, __fn__, get_timestamp
import click
import pandas as pd

logger = Logger(__fn__())


@click.command()
@click.argument('left', nargs=1, type=click.Path())
@click.argument('right', nargs=1, type=click.Path())
@click.option('--out_dir', '-o', default='data/score', type=click.Path())
def compare_results(left, right, out_dir):
    left_df = pd.read_csv(left)
    left_acc = (left_df.CLS == left_df.PRED).value_counts()[True]/len(left_df)
    logger.info(f'left acc: {left_acc:.2%}')

    right_df = pd.read_csv(right)
    right_acc = (right_df.CLS == right_df.PRED).value_counts()[True]/len(right_df)
    logger.info(f'left acc: {right_acc:.2%}')

    left_wrong = left_df.loc[left_df.PRED != left_df.CLS].copy()
    right_wrong = right_df.loc[right_df.PRED != right_df.CLS].copy()

    all_mistakes = left_wrong.join(right_wrong[['PRED', 'NEG', 'NEU', 'POS']], how='outer', rsuffix='_E3')

    tmp = all_mistakes.loc[all_mistakes.PRED_E3.isna()]
    right_improved = tmp.dropna(axis=1).join(right_df.iloc[tmp.index][['PRED', 'NEG', 'NEU', 'POS']], rsuffix='_E3')
    logger.info(f'right improved: {len(right_improved)}')

    tmp = all_mistakes.loc[all_mistakes.PRED.isna()]
    right_worse = left_df.iloc[tmp.index].join(tmp.dropna(axis=1))
    logger.info(f'right worse: {len(right_worse)}')

    fn = right.split('/')[-1].rstrip('.csv')
    right_improved.to_csv(f'{out_dir}/{fn}_improved_{get_timestamp()}.csv')
    right_worse.to_csv(f'{out_dir}/{fn}_worse_{get_timestamp()}.csv')
    logger.info(f'compare results written to {out_dir}')



if __name__ == '__main__':
    compare_results()

