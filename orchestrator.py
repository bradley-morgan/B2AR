from tools.general_tools import Obj, required_config_params as rcp
from pipelines.Selection import ModelSelectionPipe
from pipelines.Data import DataPreprocessingPipe
import os, tools.orchestration_tools as o_tools

class Orchestrator:

    # TODO Need to calculate the amount of resources to use based on parameters

    def __init__(self, config_src):

        if not os.path.isdir(config_src):
            raise NotADirectoryError(f'Provided Directories are not valid: \n'
                                     f'config src {config_src}')

        self.config_src = config_src
        self.config = None
        self.execution_chain = Obj()

        self.compile_configurations()
        self.compile_execution_chain()

    def compile_execution_chain(self):
        # validate phase 1 parameters
        required = [rcp.orch, rcp.preprocessing]

        for r in required:
            try:
                _ = self.config[r]
            except KeyError:
                raise KeyError(f'This pipeline requires the field {r} in the config')

        phases = self.config.get(rcp.orch).get(rcp.phases)
        if phases is None:
            raise KeyError(f'This pipeline requires the field phases in r')

        phase_names = [rcp.phase2, rcp.phase3, rcp.phase4]
        matched = []
        for p_name in phase_names:
            if p_name in list(phases.keys()):
                matched.append(True)

        if not any(matched):
            raise KeyError(f'This pipeline detected no valid phase name please add phases 2-4 configurations')


        chain = Obj()
        chain(phase1=Obj(pipe=DataPreprocessingPipe(self.config), output=None))

        max_cores = self.config[rcp.orch][rcp.max_cores]
        max_gpus = self.config[rcp.orch][rcp.max_gpus]
        for phase in self.config.get(rcp.orch).get(rcp.phases).keys():
            phase_config = self.config[rcp.orch][rcp.phases][phase]
            if phase == rcp.phase2:
                if not phase_config.get('active'):
                    continue

                pipe = ModelSelectionPipe(self.config, max_cores, max_gpus)
                chain(phase2=Obj(pipe=pipe, output=None))

            elif phase == rcp.phase3:
                chain(phase3=Obj(pipe='phase3-pipe', output=None))

            elif phase == rcp.phase4:
                chain(phase4=Obj(pipe='phase4-pipe', output=None))

        self.execution_chain = chain

    def compile_configurations(self):
        # check existence of config
        files = os.listdir(self.config_src)
        files = [f for f in files if f.endswith('.config.yaml')]

        if len(files) == 0:
            raise FileNotFoundError(f'No .config.yaml files found in {self.config_src}\n please provide a valid config')

        elif len(files) > 1:
            raise ValueError(f'Please provide only 1 .config.yaml file in {self.config_src}')

        files = os.path.join(self.config_src, files[0])
        self.config = o_tools.compile_yaml(files)
        self.config[rcp.orch][rcp.config_src] = self.config_src

    def execute(self):
        chain = self.execution_chain.to_list()
        chain.sort(key=lambda tup: tup[0])

        for phase, phase_obj in chain:

            if phase == rcp.phase1:
                self.execution_chain.phase1.output = phase_obj.pipe.execute()

            elif phase == rcp.phase2:
                data = self.execution_chain.phase1.output.datasets[self.config[rcp.preprocessing][rcp.output]]
                self.execution_chain.phase2.output = phase_obj.pipe.execute(data)

            elif phase == rcp.phase3:
                # TODO If phase 2 not provided then need to create a new model to feed to phase 3
                pass

            elif phase == rcp.phase4:
                # TODO If phase 2 or phase 3 not provided then need to create a new model to feed to phase 4
                pass
