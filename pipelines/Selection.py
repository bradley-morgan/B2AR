import os
import pandas as pd
import tools.orchestration_tools as o_tools
import tools.model_tools as m_tools
import multiprocessing as mp
from tools.general_tools import Obj, required_config_params as rcp
from pipes.OptimPipe import OptimPipe
import pipes.VariancePipe as VarPipe
import time
import queue as sync_q


class ModelSelectionPipe:

    def __init__(self, config: dict, max_cores: int):

        self.max_cores = max_cores
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
            top_level = Obj(
                id=idx,
                test_mode=test_mode,
                model=model,
                project=self.config[rcp.orch][rcp.project_name]
            )

            variance = Obj(
                y_true_name=self.config[rcp.variance][rcp.y_true_name],
                test_repeats=self.config[rcp.variance][rcp.test_repeats],
                n_samples=self.config[rcp.variance][rcp.n_samples],
                variance_output=self.config[rcp.variance][rcp.variance_output],
                precision_min=self.config[rcp.variance][rcp.precision][rcp.min],
                precision_max=self.config[rcp.variance][rcp.precision][rcp.max],
                repeats_output=self.config[rcp.variance][rcp.repeats_output]
            )

            validation = Obj(
                sweep_config=sweep_config,
                default_config=default_config,
                k_folds=self.config[rcp.validation][rcp.k_folds],
                n_repeats=self.config[rcp.validation][rcp.n_repeats]
            )

            chain.append(
                Obj(
                    top_level=top_level,
                    variance=variance,
                    validation=validation,
                    output=None,
                    error=None,
                )
            )

        if len(chain) == 0:
            raise ValueError('Selection Pipeline detected an empty execution chain during compilation.'
                             'The pipeline has nothing to compute')

        self.execution_chain = chain

    def config_storage_space(self):
        pass

    def feed(self, data: pd.DataFrame):
        # Feed functions defines the pipeline input dependencies
        self.data = data

    def execute(self):
        # Each pipe needs to parallise the execution chain, join the results and the return the new pipeline
        self.execution_chain = self.execute_repeat_estimation()
        self.execution_chain = self.execute_optimization()

        # TODO implement Ray or Dask Engine
        #TODO output in standardised way

    def execute_repeat_estimation(self):

        job_q = sync_q.Queue()
        for job in self.execution_chain:
            formatted_job = Obj(
                **job.top_level.to_dict(),
                **job.variance.to_dict(),
                data=self.data,
                max_cores=self.config[rcp.validation][rcp.parallelisation][rcp.max_cores]
            )
            job_q.put(formatted_job)

        while not job_q.empty():
            var_out = VarPipe.pipe(job_q)


    # TODO variance estimation
    # TODO repeats estimation
    # TODO optimisation
    # TODO cross validation
    # TODO Add tagging to pipes
    def execute_optimization(self):

        # TODO test cross validation make sure multiprocessing is faster than synchronous cross val

        timeout = self.config[rcp.selection][rcp.parallelisation][rcp.timeout] * 3600
        mode = self.config[rcp.selection][rcp.parallelisation][rcp.mode]
        cores = o_tools.get_cores(required_cores=len(self.execution_chain), max_cores=self.max_cores)

        if mode.lower() == rcp.single:
            for job in self.execution_chain:
                formatted_job = Obj(
                    **job.top_level.to_dict(),
                    **job.validation.to_dict(),
                    data=self.data,
                    max_cores=self.config[rcp.validation][rcp.parallelisation][rcp.max_cores]
                )
                p_out = OptimPipe()
                p_out.init_job(formatted_job, None, None)
                p_out.pipe()

        elif mode.lower() == rcp.multiprocessing:
            job_q = mp.Queue()
            exit_q = mp.Queue()
            processes = []
            kill_children = mp.Event()
            for _ in range(cores):
                p_out = OptimPipe()
                p = mp.Process(target=p_out.pipe, args=(job_q, kill_children, exit_q))
                p.daemon = False
                processes.append(p)
                p.start()

            for job in self.execution_chain:
                formatted_job = Obj(
                    **job.top_level.to_dict(),
                    **job.validation.to_dict(),
                    data=self.data,
                    max_cores=self.config[rcp.validation][rcp.parallelisation][rcp.max_cores]
                )
                job_q.put(formatted_job)

            s = time.time()
            time.sleep(timeout)
            print(f'TIMEOUT SWEEPS at {time.time() - s}')
            kill_children.set()
            print('Waiting for Child Processes to Finish')
            exit_array = []
            exit_flag = False
            while not exit_flag:
                while not exit_q.empty():
                    o = exit_q.get()
                    exit_array.append(o)

                if len(exit_array) == len(processes):
                    exit_flag = True

            exit_q.close()
            job_q.close()
            for p in processes:
                p.terminate()
                print('Sweep Terminated')

            print('Model Selection Pipeline Processing Complete')
            print('End Program')

    def extract_best_model(self):
        pass




