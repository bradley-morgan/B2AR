"""
VariancePipe runs bootstrap experiments calculate how much a model performance varies. This value is
then used to calculate the minimum number of repeats to run relative to desired level of precision
"""
import wandb
import services.Bootstrap as bootstrap
import tools.model_tools as m_tools

def pipe(job_q):
    """
    :param job_q: Either a multiprocessing Queue or standard synchronous queue
    """
    
    job = job_q.get()
    data = job.data
    n_repeats = job.n_repeats
    model = job.model
    project = job.project
    max_cores = job.max_cores

    # run_obj = wandb.init(
    #     project=project,
    #     config=model.params
    # )

    boot_output = []
    for repeats in range(n_repeats):

        results = bootstrap.run(data, model, repeats, max_cores)
        results = bootstrap.format_output(results)
        results = m_tools.get_descriptive_stats(results.mcc_score)
        boot_output.append(results)

    a = 0
    