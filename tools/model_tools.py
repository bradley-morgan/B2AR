from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from tools.general_tools import Obj
import numpy as np
from scipy.stats import sem


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

    def __init__(self, model: str, params: dict, test_mode: bool):

        self.model = model
        self.params = params
        self.test_mode = test_mode

    def make(self):
        if 'decision tree' == self.model.lower() or 'decision_tree' or self.model.lower() \
                or 'd_tree' == self.model.lower() or 'd tree' == self.model.lower() or 'dt' == self.model.lower():

            return self.is_test(DecisionTreeClassifier)

        raise ValueError(f'Make Model Function detected an invalid model type {self.model}')

    def is_test(self, model):
        if self.test_mode:
            return model()
        else:
            return model(**self.params)