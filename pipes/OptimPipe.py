import wandb
import services.CrossValidation as cross_val
import pandas as pd
from tools.general_tools import Obj, required_config_params as rcp
import tools.model_tools as m_tools
from multiprocessing import Event


class OptimPipe:

    def __init__(self, job: Obj):

        self.data = job.data
        self.k_folds = job.k_folds
        self.n_repeats = job.n_repeats
        self.model = job.model
        self.project = job.project
        self.sweep_config = job.sweep_config
        self.max_cores = job.max_cores

    def process(self):
        # TODO Init run with sweep config
        run_obj = wandb.init(
            config=self.model.params,
            project=self.project
        )
        config = wandb.config
        self.model.params = config._items

        # TODO run cross validation
        # cross_val = CrossValService(self.k_folds, 1, self.data, self.model)
        results = cross_val.run(
            data=self.data,
            k_folds=self.k_folds,
            n_repeats=self.n_repeats,
            model=self.model,
            max_cores=self.max_cores,
        )
        results = cross_val.get_descriptives(results.cross_val_mcc_scores)
        # Log target metric to cloud
        run_obj.log({'mean_mcc': results.mean})
        # save run name to file

    def run(self):
        sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project)
        wandb.agent(sweep_id, function=self.process)

