import typing
import os
from . import models
from .model_base import DATA_DIR, DATA_PATH
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_data():
    if not os.path.isdir(DATA_PATH):
        import urllib.request
        import tarfile
        urllib.request.urlretrieve('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', 'data.tar.gz')
        tarfile.open('data.tar.gz').extractall(path=DATA_DIR)
        os.remove('data.tar.gz')
    for name, model in models.models.items():
        logging.info('Training {}...'.format(name))
        model.fit()
        logging.info('Done')

def analize_text(text: str, model: str) -> typing.Tuple[int, int]:
    if model not in models.models:
        raise ValueError('Invalid model {}, must be one from {}'.format(model, models.models.keys()))
    score = int(models.models[model].fit_predict_str(text))
    return score, score > 5

load_data()
