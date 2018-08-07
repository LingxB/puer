from src.models import ATLSTM
from src.utils import Logger, __fn__, load_corpus, list_files
import click



VALID_MODELS = dict(
    atlstm=ATLSTM
)

logger = Logger(__fn__())


@click.command()
@click.argument('model_dir', nargs=1, type=click.Path())
@click.argument('test_files', nargs=-1, type=click.Path())
def test(model_dir, test_files):

    # Get model name
    model_files = list_files(model_dir)
    model_name = list(set([n.split('.')[0] for n in model_files if n.split('.')[-1] in {'index','meta'}]))
    assert len(model_name) == 1, 'Multiple model names in model directory!'
    model_name = model_name[0]
    logger.info('Found model {} in given directory'.format(model_name.lower()))

    # Load model
    model = VALID_MODELS[model_name]
    model = model()
    model.load(model_dir)

    # Load data
    test_df = load_corpus(list(test_files))

    # Score
    _, _, loss_, acc3_ = model.score(test_df)
    logger.info('test_loss={loss:.4f} ' \
                'test_acc3={acc:.2%}'\
                .format(loss=loss_, acc=acc3_))


if __name__ == '__main__':
    test()