from tools.general_tools import required_config_params as rcp
from services.Data import DataService


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, data_service: DataService):

        keys = data_service.datasets.keys()
        search_params = self.config[rcp.search_params]

        for key in keys:
            data = data_service.datasets[key][rcp.data]
            cols = []
            for col in data.columns:
                finds = []
                for search_param in search_params:
                    if search_param not in col:
                        finds.append(False)
                    else:
                        finds.append(True)

                if not any(finds):
                    cols.append(col)

            data_service.datasets[key][rcp.data] = data[cols]

        return data_service