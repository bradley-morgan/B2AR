import wandb
from services.CrossValidation import CrossValService
import pandas as pd
from tools.general_tools import Obj, required_config_params as rcp
import tools.model_tools as m_tools


class OptimPipe:

    def __init__(self, data: pd.DataFrame, k_folds: int, n_repeats: int,
                 sweep_config: dict, model: m_tools.MakeModel, project: str):
        self.data = data
        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.model = model
        self.project = project
        self.sweep_config = sweep_config
        self.run_obj = None

    def process(self):
        # TODO Init run with sweep config
        run_obj = wandb.init(
            config=self.model.params,
            project=self.project
        )
        config = wandb.config
        self.model.params = config

        # TODO run cross validation
        cross_val = CrossValService(self.k_folds, 1, self.data, self.model)
        cross_val.run()
        results = cross_val.get_descriptives()
        # Log target metric to cloud
        run_obj.log({'mean_mcc': results.mean})
        # save run name to file


    def run(self):
        sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project)
        wandb.agent(sweep_id, function=self.process)

