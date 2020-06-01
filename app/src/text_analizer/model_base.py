import pandas as pd
import pathlib
import os

SavedDataCache = {}
DATA_DIR = 'sentiment_data'
DATA_PATH = DATA_DIR + '/aclImdb'

class ModelBase:
    def __init__(self) -> None:
        pass

    def data_reader(self, dir_name: str) -> None:
        for entry in os.scandir(dir_name / 'pos'):
            # Maybe regex would be better, but this is OK
            # file format: id_score.txt
            # example: 1234_2.txt
            score = entry.name.split('_')[1].split('.')[0]
            text = open(entry.path).read()
            yield {'text': text, 'truth': score}
        for entry in os.scandir(dir_name / 'neg'):
            # Maybe regex would be better, but this is OK
            # file format: id_score.txt
            # example: 1234_2.txt
            score = entry.name.split('_')[1].split('.')[0]
            text = open(entry.path).read()
            yield {'text': text, 'truth': score}


    def read_data(self, dir_name: str, lower_case: bool=True) -> pd.DataFrame:
        if dir_name in SavedDataCache:
            return SavedDataCache[dir_name]
        currPath = pathlib.Path(DATA_PATH) / dir_name
        df = pd.DataFrame(self.data_reader(currPath), columns=['text', 'truth'])

        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')

        # Lowercase seems to be good idea
        if lower_case:
            df['text'] = df['text'].str.lower()
        SavedDataCache[dir_name] = df
        return df

    def confusion_matrix(self, df: pd.DataFrame, normalize: str='true') -> None:
        cm = confusion_matrix(df['truth'], df['pred'], normalize=normalize, labels=list(range(1, 11, 1)))
        classes = range(1, 11, 1)
        fig, ax = plt.subplots(figsize=(20,20))
        im = ax.imshow(cm, interpolation='nearest', origin='lower', cmap=plt.cm.YlOrBr)
        ax.figure.colorbar(im, ax=ax)
        # Show all ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # Label with respective list entries
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')

        # Set alignment of tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        return fig, ax

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))

