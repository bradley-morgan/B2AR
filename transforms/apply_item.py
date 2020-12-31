from tools.general_tools import required_config_params as rcp
from services.Data import DataService
import pandas as pd


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, data: DataService):

        if self.config.keys() is None:
            return data

        for item in self.config.keys():
            params = self.config[item]
            name = params[rcp.name]
            item_to_apply = params[rcp.item]
            apply_to = params[rcp.apply_to]

            for dataset_name in apply_to:
                data.datasets[dataset_name][rcp.data][name] = pd.Series([item_to_apply] * len(data.datasets[dataset_name][rcp.data]))

        return data