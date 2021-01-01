from tools.general_tools import required_config_params as rcp
from services.Data import DataService


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, data: DataService):

        value = self.config[rcp.value]
        datasets = data.datasets

        keys = list(datasets.keys())

        for name in keys:
            datasets[name][rcp.data].fillna(value, inplace=True)
            nan_count = datasets[name][rcp.data].isna().sum().sum()
            if nan_count > 0:
                raise ValueError(f'Transform Error: Change nans has failed. transform detected {nan_count} NaNs Remaining')

        data.datasets = datasets
        return data