from tools.general_tools import Obj, required_config_params as rcp
import tools.general_tools as g_tools
import os, pandas as pd
from sklearn.model_selection import train_test_split

"""
Dataset class acts as a single source of truth for all data and is consumed by Model Objects.

Expected Format for Datasets
    Each datasets are expected to csvs
    Datasets will be names - dataset Name/dataset component name 
    Datasets can be organised as follows:
            - root
                - Dataset name 1
                        - Dataset components (csv files only)
                - Dataset name 2
                        - Dataset components

Constructor Arguments
    src - List of file paths to dataset directories
    transforms = List of functions used to transform datasets
"""


class DataService:

    def __init__(self, config):
        self.config = config
        self.datasets = {}

    def load(self):
        src = g_tools.path(self.config[rcp.orch][rcp.src])
        for sub_folder in os.listdir(src):
            sub_file_list = os.listdir(os.path.join(src, sub_folder))
            sub_file_list = [file for file in sub_file_list if file.endswith('.csv')]
            for file in sub_file_list:
                data = pd.read_csv(os.path.join(src, sub_folder, file))
                dataset_name = f'{sub_folder}-{file.split(".")[0]}'
                self.datasets[dataset_name] = {rcp.data: data}

    def provide(self, name, dtype):
        y_name = self.config[rcp.preprocessing][rcp.target_name]
        dataFrame = self.datasets[name][rcp.data]
        y = dataFrame[y_name].to_numpy()
        x = dataFrame.drop(y_name, axis=1)
        feature_names = x.columns.to_numpy()
        x = x.to_numpy()

        if dtype:
            x = x.astype(dtype)
            y = y.astype(dtype)

        test_size = self.config[rcp.preprocessing][rcp.test_size]
        x_train, x_hold_out, y_train, y_hold_out = train_test_split(x, y,
                                                            test_size=test_size,
                                                            shuffle=True,
                                                            stratify=y)

        return Obj(
                    x_train=x_train,
                    x_hold_out=x_hold_out,
                    y_train=y_train,
                    y_hold_out=y_hold_out,
                    feature_names=feature_names
        )