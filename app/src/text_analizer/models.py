from .linear_regression import LogisticRegressionSentiment, LogisticRegressionSentimentSum
from .svm import SVMSentiment, SVMSentimentSum

models = {
        'logistic': LogisticRegressionSentiment(),
        'logistic_sum': LogisticRegressionSentimentSum(),
        'svm': SVMSentiment(),
        'svm_sum': SVMSentimentSum(),
    }

