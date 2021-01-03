import os
import pandas as pd
import tools.orchestration_tools as o_tools
import tools.model_tools as m_tools
from multiprocessing import Process, freeze_support
from tools.general_tools import Obj, required_config_params as rcp
from services.Data import DataService
from pipes.OptimPipe import OptimPipe
import wandb


class ModelSelectionPipe:

    def __init__(self, config: Obj, max_cores: int, max_gpus: int):

        self.max_cores = max_cores
        self.max_gpus = max_gpus
        self.config = config

        self.execution_chain = []
        self.data = None
        self.compile_execution_chain()

    def compile_execution_chain(self):

        chain = []
        config_src = self.config[rcp.orch][rcp.config_src]
        sweep_src = os.path.join(config_src, self.config[rcp.selection][rcp.sweep_src])
        # check existence of config
        if not os.path.isdir(sweep_src):
            raise NotADirectoryError(f'The sweep src {sweep_src} is not directory.'
                                     f' Sweep config folders should contain .sweep.yaml files')

        dirs = os.listdir(sweep_src)

        if len(dirs) == 0:
            raise FileNotFoundError(
                f'No files or directories found in {sweep_src}\n please provide a valid config')

        for idx, d in enumerate(dirs):
            files = os.listdir(os.path.join(sweep_src, d))
            if len(files) == 0:
                raise FileNotFoundError(
                    f'No files or directories found in {d}\n please provide a valid config')
            elif len(files) > 2 or len(files) < 2:
                raise FileExistsError(
                    f'Each sweep config should be directory containing 1 .default.yaml and 1 .sweep.yaml')

            s_files = 0
            d_files = 0
            sweep_config_path = None
            default_config_path = None
            for f in files:
                if f.endswith('.sweep.yaml'):
                    s_files += 1
                    sweep_config_path = f
                elif f.endswith('.default.yaml'):
                    d_files += 1
                    default_config_path = f

            if s_files == 0 or s_files > 1:
                raise FileExistsError(f'Found {s_files} .sweep.yaml files in {d}. each sweep config directory'
                                      f' should contain 1 .sweep.yaml file')

            if d_files == 0 or d_files > 1:
                raise FileExistsError(f'Found {d_files} .default.yaml files in {d}. each sweep config directory'
                                      f' should contain 1 .default.yaml file')

            sweep_config = o_tools.compile_yaml(os.path.join(sweep_src, d, sweep_config_path))
            default_config = o_tools.compile_yaml(os.path.join(sweep_src, d, default_config_path))

            if default_config is None:
                raise ValueError(f'Default Config is empty in {default_config_path}')

            if sweep_config is None:
                raise ValueError(f'Default Config is empty in {sweep_config_path}')

            model_name = sweep_config['model']
            parameters = sweep_config['parameters']

            if parameters is None:
                raise KeyError(f'Sweep configs must specify a non-empty parameter value')

            test_mode = self.config[rcp.orch][rcp.test_mode]
            model = m_tools.MakeModel(model_name, default_config, test_mode)
            chain.append(
                Obj(
                    id=idx,
                    model=model,
                    sweep_config=sweep_config,
                    default_config=default_config,
                    output=None,
                    error=None,
                )
            )

        if len(chain) == 0:
            raise ValueError('Selection Pipeline detected an empty execution chain during compilation.'
                             'The pipeline has nothing to compute')

        self.execution_chain = chain

    def __update_exe_chain__(self, p_out: Obj):
        self.execution_chain[p_out.id].output = p_out.output
        self.execution_chain[p_out.id].status = p_out.status
        self.execution_chain[p_out.id].error = p_out.error

    def execute(self, data: pd.DataFrame):
        self.data = data

        # Each pipe needs to parallise the execution chain, join the results and the return the new pipeline
        self.execution_chain = self.execute_repeat_estimation()
        self.execution_chain = self.execute_optimization()

        # TODO implement Ray or Dask Engine
        #TODO output in standardised way

    def execute_repeat_estimation(self):

        return self.execution_chain
        # TODO variance estimation
        # TODO repeats estimation
        # TODO optimisation
        # TODO cross validation

    def execute_optimization(self):

        parallelisation = self.config[rcp.selection][rcp.parallelisation]
        project = self.config[rcp.orch][rcp.project_name]
        k_folds = self.config[rcp.validation][rcp.k_folds]
        n_repeats = self.config[rcp.validation][rcp.n_repeats]

        if parallelisation.lower() == rcp.single:
            for job in self.execution_chain:
                p_out = OptimPipe(self.data, k_folds, n_repeats, job.sweep_config, job.model, project).run()
                self.__update_exe_chain__(p_out)

        elif parallelisation.lower() == rcp.multiprocessing:
            import time
            timeout = self.config[rcp.selection][rcp.timeout] * 3600
            processes = []
            start = []
            for job in self.execution_chain:
                pipe = OptimPipe(self.data, k_folds, n_repeats, job.sweep_config, job.model, project)
                process = Process(target=pipe.run)
                process.daemon = False
                processes.append(process)
                start.append(time.time())
                process.start()
                a = 0

            for process, s in zip(processes, start):
                process.join(timeout=timeout)
                process.terminate()
                end = time.time() - s
                print(f'process timed out at: {end}')

