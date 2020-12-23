import tools.orchestration_tools as o_tools
import argparse

parser = argparse.ArgumentParser(description='placeholder description')
parser.add_argument('--config_src', '-csrc', type=str, default='./configs/exp1', help='point to configuration folder')

args = parser.parse_args()

orchestrator = o_tools.Orchestrator(config_src=args.config_src)
