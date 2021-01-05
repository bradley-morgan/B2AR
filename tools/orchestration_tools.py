import os
import yaml

# ==================================== TODO FUNCTIONS ==================================================================


def get_cores(required_cores, max_cores=-1):
    total_cores = os.cpu_count()

    if max_cores == -1:
        return total_cores

    elif required_cores <= max_cores:
        return required_cores

    elif required_cores > max_cores:

        if max_cores <= total_cores:
            return max_cores
        else:
            raise OSError(f'The number of max cores to use {max_cores} exceeds the total availble cores {total_cores}'
                          f'on this machine')


def compile_yaml(path):
    with open(path, mode='r') as yam_file:
         config = yaml.load(yam_file, Loader=yaml.FullLoader)
    return config

# ==================================== TODO CLASSES ====================================================================

