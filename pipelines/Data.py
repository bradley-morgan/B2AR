from tools.general_tools import Obj, required_config_params as rcp
import tools.general_tools as g_tools
from services.Data import DataService
import os, importlib

class DataPreprocessingPipe:

    def __init__(self, config: Obj):
        self.config = config
        self.transform_src = g_tools.path(self.config[rcp.preprocessing][rcp.transform_src])
        self.execution_chain = []
        self.transform_src = g_tools.path('transforms')

        self.compile_execution_chain()

    def compile_execution_chain(self):
        """
        Checks that user-set transforms exist as files in the transform folder
        If it exists it then load and initializes the transform
        :return:
        """
        transform_files = os.listdir(self.transform_src)
        transform_files = [file for file in transform_files if file.endswith('.py')]

        user_transforms = self.config[rcp.preprocessing][rcp.transforms]

        transforms = []
        for transform in user_transforms.keys():
            if f'{transform}.py' in transform_files:

                plugin = importlib.import_module(f'transforms.{transform}')
                plugin = plugin.Transform(user_transforms[transform])
                transforms.append(plugin)

        self.execution_chain = transforms

    def execute(self):
        # Execute transforms
        data = DataService(self.config)
        data.load()
        for transform in self.execution_chain:
            data = transform(data)

        # TODO Output data in standardised way
        return data