import os
import pandas as pd
from sklearn import tree
import wandb
import os
import matplotlib.pyplot as plt
import subprocess
import matplotlib


def path(path:str) -> str:
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    f_path = os.path.join(MAIN_DIR, path)
    return f_path


class Obj:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def compare_attributes(self, other):
        return self.__dict__ == other.__dict__

    def __call__(self, **kwds):
        self.__dict__.update(kwds)

    def iter(self):
        a = self.__dict__.items()
        return a

    def exist(self, item):
        return self.__dict__.get(item) is not None

    def delete(self, item):
        if isinstance(item, list):
            for i in item:
                del self.__dict__[i]
        else:
            del self.__dict__[item]

    def to_dict(self):
        return self.__dict__

    def to_list(self):
        return list(self.iter())

    def to_dataframe(self, orient: str, columns=None):
        """
        Converts Anonymous object to a pandas dataframe
        :param orient: Should Object Atrribute names be the 'columns' or the 'index' of the dataframe
        :param columns: 'Only use if orient is 'index', represents the column names
        :return: Dataframe
        """
        return pd.DataFrame.from_dict(self.to_dict(), orient=orient, columns=columns)


class ImageSaver:

    def __init__(self, run):
        matplotlib.use('Agg')
        self.save_dir = path('../tmp')
        self.run = run

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.clean_up()

    def save(self, plot, name: str, format: str):

        save_path = os.path.join(self.save_dir, f'{name}.{format}')
        #TODO This needs Testing
        plot.savefig(save_path, format=format, dpi=300)
        self.run.log({name: wandb.Image(save_path)})
        self.clean_up()

    def save_graphviz(self, model: tree.DecisionTreeClassifier,
                      feature_names: list,
                      class_names: list,
                      graph_name: str,):

        name = 'tree_graph'
        format = 'dot'

        dot_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        tree.export_graphviz(
            model,
            out_file=dot_out_file,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
        )
        # Convert to png
        format = 'png'
        png_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        out = subprocess.run(['dot', '-Tpng', dot_out_file, '-o', png_out_file])

        self.run.log({graph_name: wandb.Image(png_out_file)})

        if out.returncode != 0:
            raise ValueError('ImageSave.save_graphviz: Graphviz dot to png command failed during subprocess run')

    def clean_up(self):
        plt.clf()
        # Clear tmp folder of files no longer needed
        files = os.listdir(self.save_dir)
        for file in files:
            os.remove(os.path.join(self.save_dir, file))


required_config_params = Obj(
    orch='orch',
    preprocessing='preprocessing',
    variance='variance',
    repeats='repeats',
    validation='validation',
    selection='selection',
    uncertainty='uncertainty',
    inspection='inspection',
    src='src',
    project_name='project_name',
    cloud='cloud',
    enable_cloud_logging='enable_cloud_logging',
    api_key='api_key',
    test_mode='test_mode',
    max_cores='max_cores',
    max_gpus='max_gpus',
    phases='phases',
    phase1='phase1',
    phase2='phase2',
    phase3='phase3',
    phase4='phase4',
    active='active',
    inputs='inputs',
    model='model',
    test_size='test_size',
    transforms='transforms',
    test_repeats='test_repeats',
    n_samples='n_samples',
    time='time',
    units='units',
    threshold='threshold',
    std_threshold='std_threshold',
    output='output',
    parallelisation='parallelisation',
    single='single',
    multiprocessing='multiprocessing',
    mode='mode',
    timeout='timeout',
    timeout_children='timeout_children',
    precision='precision',
    min='min',
    max='max',
    k_folds='k_folds',
    n_repeats='n_repeats',
    sweep_src='sweep_src',
    performance_threshold='performance_threshold',
    correlation='correlation',
    target_name='target_name',
    data='data',
    transform_src='transform_src',
    config_src='config_src',
    exceptions='exceptions',
    best_performance='best_performance',
    merge_all='merge_all',
    merge_all_name='merge_all_name',
    groups='groups',
    group_names='group_names',
    search_params='search_params',
    remove_feature='remove_features',
    change_nans='change_nans',
    value='value',
    name='name',
    apply_to='apply_to',
    item='item',
    kill_event='kill_event'
)