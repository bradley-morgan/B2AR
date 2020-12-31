from tools.general_tools import required_config_params as rcp
from services.Data import DataService


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, dataService: DataService):
        datasets = dataService.datasets

        merge_all = self.config[rcp.merge_all]
        merge_all_name = self.config[rcp.merge_all_name]
        groups = self.config[rcp.groups]
        group_names = self.config[rcp.group_names]

        out_dataset = {}
        if len(groups) > 0:
            for group, group_name in zip(groups, group_names):
                group = list(group)
                f_key = group.pop()
                merged_df = datasets[f_key][rcp.data]

                for df_name in group:
                    subset = datasets[df_name][rcp.data]
                    merged_df = merged_df.append(subset, sort=False)

                out_dataset[group_name] = {rcp.data: merged_df}
        else:
            out_dataset = datasets

        if merge_all:
            j_key = group_names.pop()
            merged_df = out_dataset[j_key][rcp.data]
            for name in group_names:
                subset = out_dataset[name][rcp.data]
                merged_df = merged_df.append(subset, sort=False,)

            out_dataset[merge_all_name] = {rcp.data: merged_df}

        dataService.datasets = out_dataset
        return dataService
