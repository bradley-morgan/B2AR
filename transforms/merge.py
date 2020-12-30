from tools.general_tools import required_config_params as rcp
from services.Data import DataService


class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, data: DataService):
        datasets = data.datasets

        merge_all = self.config[rcp.merge_all]
        merge_all_name = self.config[rcp.merge_all_name]
        groups = self.config[rcp.groups]
        group_names = self.config[rcp.group_names]

        out_dataset = {}
        if len(groups) > 0:
            for group, group_name in zip(groups, group_names):
                group = list(group)
                f_key = group.pop()
                merged_df = datasets[f_key]["data"]

                for df_name in group:
                    data = datasets[df_name]["data"]
                    merged_df = merged_df.append(data, sort=False)

                out_dataset[group_name] = {'data': merged_df}
        else:
            out_dataset = datasets

        if merge_all:
            j_key = group_names.pop()
            merged_df = out_dataset[j_key]["data"]
            for name in group_names:
                data = out_dataset[name]["data"]
                merged_df = merged_df.append(data, sort=False,)

            out_dataset[merge_all_name] = {'data': merged_df}

        data.datasets = out_dataset
        return data
