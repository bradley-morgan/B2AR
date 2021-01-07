import tools.orchestration_tools as o_tools
from multiprocessing import Pool
import pandas as pd

def generate(data, repeats):
    # Convert data to dataframes so that we can preserve the indexes. This will allow to extract the out of bag test
    data = pd.DataFrame(data.x_train, columns=data.feature_names)
    data['target'] = data.y_train

def execute():
    pass

def format_output(data: list):
    pass

def run(data, model, repeats, max_cores):

    boostraps = generate(data, repeats)
    cores = o_tools.get_cores(required_cores=len(boostraps), max_cores=max_cores)

    with Pool(processes=cores) as pool:
        results = pool.map(execute, boostraps)

    print('Pool Terminated')
    return results

