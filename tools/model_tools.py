from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from tools.general_tools import Obj
import numpy as np
from scipy.stats import sem
import seaborn as sns
import matplotlib.pyplot as plt


def get_median_confusion_matrix(conf_mat: list):
    mat = np.asarray(conf_mat)
    median_confusion_matrix = np.median(mat, axis=0)
    return median_confusion_matrix.astype('int64')


def plot_confusion_matrix(conf_mat):
    ax = sns.heatmap(conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.close(ax.get_figure())
    plot = ax.get_figure()
    return plot


def get_descriptive_stats(scores):
    return Obj(
            mean=np.mean(scores),
            median=np.median(scores),
            std=np.std(scores),
            ste=sem(scores)
        )


def get_model_performance(y_true, y_preds):
    conf_mat = confusion_matrix(y_true, y_preds)
    mcc_score = matthews_corrcoef(y_true=y_true, y_pred=y_preds)

    return Obj(
        conf_mat=conf_mat,
        mcc_score=mcc_score,
    )


class MakeModel:

    def __init__(self, model_type: str, params: dict, test_mode: bool):

        self.model_type = model_type
        self.params = params
        self.test_mode = test_mode

    def make(self):
        if 'decision tree' == self.model_type.lower() or 'decision_tree' or self.model_type.lower() \
                or 'd_tree' == self.model_type.lower() or 'd tree' == self.model_type.lower() or 'dt' == self.model_type.lower():

            return self.is_test(DecisionTreeClassifier)

        raise ValueError(f'Make Model Function detected an invalid model type {self.model_type}')

    def is_test(self, model):
        if self.test_mode:
            return model()
        else:
            return model(**self.params)
