from sklearn.model_selection import StratifiedKFold
from tools.general_tools import Obj
from tools.model_tools import get_model_performance
import tools.model_tools as m_tools


class CrossValService:

    def __init__(self, k_folds, n_repeats, data, make_model: m_tools.MakeModel):

        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.model_make = make_model
        self.data = data
        self.results = None

    def run(self):
        # Cross validation
        cross_val_mcc_scores = []
        cross_val_conf_matrices = []

        for r in range(self.n_repeats):
            r = 1 if r == 0 else r

            cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
            for train_idx, test_idx in cv.split(self.data.x_train, self.data.y_train):
                # extract hold out test set
                train_x, val_x = self.data.x_train[train_idx], self.data.x_train[test_idx]
                train_y, val_y = self.data.y_train[train_idx], self.data.y_train[test_idx]

                # Fit & Cross validate
                cross_val_model = self.model_make.make()
                cross_val_model.fit(train_x, train_y)
                y_preds = cross_val_model.predict(val_x)

                cross_val_scores = get_model_performance(y_true=val_y, y_preds=y_preds)

                cross_val_mcc_scores.append(cross_val_scores.mcc_score)
                cross_val_conf_matrices.append(cross_val_scores.conf_mat)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = self.model_make.make()

        hold_out_model.fit(self.data.x_train, self.data.y_train)
        hold_out_y_preds = hold_out_model.predict(self.data.x_hold_out)
        hold_out_scores = get_model_performance(y_true=self.data.y_hold_out, y_preds=hold_out_y_preds)

        self.results = Obj(
            cross_val_mcc_scores=cross_val_mcc_scores,
            cross_val_conf_matrices=cross_val_conf_matrices,
            hold_out_mcc_score=hold_out_scores.mcc_score,
            hold_out_conf_matrice=hold_out_scores.conf_mat,
            model=hold_out_model
        )

    def get_descriptives(self):
        return m_tools.get_descriptive_stats(self.results.cross_val_mcc_scores)
