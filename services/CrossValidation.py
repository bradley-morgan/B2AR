from sklearn.model_selection import StratifiedKFold
from tools.general_tools import Obj
from tools.model_tools import get_model_performance
import tools.model_tools as m_tools
from multiprocessing import Pool, Queue, Process, current_process
import tools.orchestration_tools as o_tools
import wandb


def generate(data: Obj, model, k_folds: int, n_repeats: int):
    folds = []
    for r in range(n_repeats):
        r = 1 if r == 0 else r

        cv = StratifiedKFold(n_splits=k_folds, shuffle=True)
        for train_idx, test_idx in cv.split(data.x_train, data.y_train):
            # extract hold out test set
            train_x, val_x = data.x_train[train_idx], data.x_train[test_idx]
            train_y, val_y = data.y_train[train_idx], data.y_train[test_idx]
            folds.append(Obj(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                model=model
            ))
    return folds


def execute(fold):
    # Fit & Cross validate
    print(f'Spawning Child in {current_process()}')
    cross_val_model = fold.model.make()
    cross_val_model.fit(fold.train_x, fold.train_y)
    y_preds = cross_val_model.predict(fold.val_x)

    cross_val_scores = get_model_performance(y_true=fold.val_y, y_preds=y_preds)
    return cross_val_scores


def run(data, k_folds, n_repeats, model, max_cores):
    # Cross validation

    k_folds = generate(data, model, k_folds, n_repeats)
    cores = o_tools.get_cores(required_cores=len(k_folds), max_cores=max_cores)

    with Pool(processes=cores) as pool:
        results = pool.map(execute, k_folds)

    print('Pool Terminated')
    return results


def format_output(data: list):
    conf_mat = []
    mcc_score = []
    for obj in data:
        conf_mat.append(obj.conf_mat)
        mcc_score.append(obj.mcc_score)

    return Obj(
        conf_mat=conf_mat,
        mcc_score=mcc_score
    )



