from sklearn.tree import DecisionTreeClassifier


def make_model(model: str, params: dict):

    if 'decision tree' == model.lower() or 'decision_tree' or model.lower() \
            or 'd_tree' == model.lower() or 'd tree' == model.lower() or 'dt' == model.lower():

        return DecisionTreeClassifier(**params)

    raise ValueError(f'Make Model Function detected an invalid model type {model}')
