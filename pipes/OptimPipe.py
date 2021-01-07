import wandb
import services.CrossValidation as cross_val
import pandas as pd
import tools.general_tools as g_tools
import tools.model_tools as m_tools
from multiprocessing import Queue, Event, current_process


class OptimPipe:

    def __init__(self):

        self.data = None
        self.k_folds = None
        self.n_repeats = None
        self.model = None
        self.project = None
        self.sweep_config = None
        self.max_cores = None
        self.kill_self = None
        self.exit_q = None
        self.freeze_loop = False

    def init_job(self, job, kill_event, exit_queue):
        self.data = job.data
        self.k_folds = job.k_folds
        self.n_repeats = job.n_repeats
        self.model = job.model
        self.project = job.project
        self.sweep_config = job.sweep_config
        self.max_cores = job.max_cores
        self.kill_self = kill_event
        self.exit_q = exit_queue

    def is_pipe_gate_closed(self):
        if self.kill_self is not None:
            if self.kill_self.is_set():
                if not self.freeze_loop:
                    print('Agent sees that kill is True: Pipe Gate Closed \n')
                    self.exit_q.put(True)
                    self.freeze_loop = True
                return True
            else:
                print('Agent sees that kill self is False: Pipe Gate Open')
                return False
        else:
            print('Pipe is running  synchronously: Pipe Gate Open')
            return False

    def process(self):

        if self.is_pipe_gate_closed():
            return

        # TODO Init run with sweep config
        run_obj = wandb.init(
            config=self.model.params,
            project=self.project
        )
        config = wandb.config
        self.model.params = config._items
        image_saver = g_tools.ImageSaver(run_obj)

        # TODO run cross validation
        # cross_val = CrossValService(self.k_folds, 1, self.data, self.model)
        results = cross_val.run(
            data=self.data,
            k_folds=self.k_folds,
            n_repeats=self.n_repeats,
            model=self.model,
            max_cores=self.max_cores,
        )
        results = cross_val.format_output(results)
        mcc_score = m_tools.get_descriptive_stats(results.mcc_score)

        # Log target metric to cloud
        run_obj.log({'mean_mcc': mcc_score.mean})
        image_saver.save(
            m_tools.plot_confusion_matrix(m_tools.get_median_confusion_matrix(results.conf_mat)),
            f'{current_process().name}-{run_obj.name}-confusion-matrix',
            'png'
        )
        # save run name to file

    def pipe(self, job_q: Queue = None, kill_event: Event = None, exit_q: Queue = None):

        if job_q is not None:
            job = job_q.get()
            self.init_job(job, kill_event, exit_q)

        sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project)
        wandb.agent(sweep_id, function=self.process)




