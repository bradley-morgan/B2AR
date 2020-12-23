import os, stat
import yaml
# ==================================== TODO FUNCTIONS ==================================================================


def compile_yaml(path):
    with open(path, mode='r') as yam_file:
         config = yaml.load(yam_file, Loader=yaml.FullLoader)
    return config

# ==================================== TODO CLASSES ====================================================================
class Orchestrator:

    def __init__(self, config_src):

        if not os.path.isdir(config_src):
            raise NotADirectoryError(f'Provided Directories are not valid: \n'
                                     f'config src {config_src}')

        self.config_src = config_src
        self.config = None

        self.compile_configurations()

    def compile_configurations(self):
        # check existence of config
        files = os.listdir(self.config_src)
        files = [f for f in files if f.endswith('.config.yaml')]

        if len(files) == 0:
            raise FileNotFoundError(f'No .config.yaml files found in {self.config_src}\n please provide a valid config')

        elif len(files) > 1:
            raise ValueError(f'Please provide only 1 .config.yaml file in {self.config_src}')

        files = os.path.join(self.config_src, files[0])
        self.config_src = compile_yaml(files)

    def run(self):
        pass
