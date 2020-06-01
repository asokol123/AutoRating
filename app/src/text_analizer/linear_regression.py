from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .model_base import ModelBase
import pandas as pd
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class LogisticRegressionSentiment(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline(
            [
                ('Vectorize', CountVectorizer()),
                ('TF-IDF', TfidfTransformer()),
                ('Classify', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )
        self.learner = None

    def fit(self, train_file: str='train', lower_case: bool=True) -> None:
        train_df = self.read_data(train_file, lower_case)
        self.learner = self.pipeline.fit(train_df['text'], train_df['truth'])

    def predict(self, test_file: str='test', lower_case: bool=True) -> pd.DataFrame:
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = self.learner.predict(test_df['text'])
        return test_df

    def fit_predict(
            self,
            train_file: str='train',
            test_file: str='test',
            lower_case: bool=True,
            refit: bool=False) -> pd.DataFrame:
        if self.learner is None or refit:
            self.fit(train_file, lower_case)
        return self.predict(test_file, lower_case)

    def predict_str(self, text: str) -> int:
        return self.learner.predict([text])[0]

    def fit_predict_str(
            self,
            test_text: str,
            train_file: str='train',
            lower_case: bool=True,
            refit: bool=False) -> int:
        if self.learner is None or refit:
            logging.info('learner: {}'.format(self.learner))
            self.fit(train_file, lower_case)
        return self.predict_str(test_text)


class LogisticRegressionSentimentSum(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline(
            [
                ('Vectorize', CountVectorizer()),
                ('TF-IDF', TfidfTransformer()),
                ('Classify', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )
        self.learner = None

    def fit(self, train_file: str='train', lower_case: bool=True) -> None:
        train_df = self.read_data(train_file, lower_case)
        self.learner = self.pipeline.fit(train_df['text'], train_df['truth'])

    def predict(self, train_file: str='train', test_file: str='test', probs_pow: float=2.5, lower_case: bool=True) -> pd.DataFrame:
        test_df = self.read_data(test_file, lower_case)
        probs = self.learner.predict_proba(test_df['text'])
        probs = probs ** probs_pow
        probs /= probs.sum(axis=1)[:, np.newaxis]
        test_df['pred'] = np.rint(np.sum(probs * learner.classes_, axis=1)).astype(int)
        return test_df

    def fit_predict(
            self,
            train_file: str='train',
            test_file: str='test',
            refit: bool=False,
            probs_pow: float=2.5,
            lower_case: bool=True) -> pd.DataFrame:
        if self.learner is None or refit:
            self.fit(train_file, lower_case)
        return self.predict(test_file, probs_pow, lower_case)

    def predict_str(self, text: str, probs_pow: float=2.5, lower_case: bool=True) -> int:
        if lower_case:
            text = text.lower()
        probs = self.learner.predict_proba([text])
        probs = probs ** probs_pow
        probs /= probs.sum(axis=1)[:, np.newaxis]
        result = np.rint(np.sum(probs * self.learner.classes_, axis=1)).astype(int)
        return result[0]

    def fit_predict_str(
            self,
            test_text: str,
            probs_pow: float=2.5,
            train_file: str='train',
            lower_case: bool=True,
            refit: bool=False) -> int:
        if self.learner is None or refit:
            self.fit(train_file, lower_case)
        return self.predict_str(test_text, probs_pow, lower_case)

