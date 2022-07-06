from elliot.run import run_experiment
import argparse
'''
parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--model', type=str, default='egcfv2')
parser.add_argument('--dataset', type=str, default='facebook_exploration_l2_e32')
args = parser.parse_args()

run_experiment(f"config_files/{args.model}/{args.dataset}.yml")

from elliot.run import run_experiment
import argparse
'''

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--config', type=str, default='egcfv2/facebook_exploration_l2_e32')
args = parser.parse_args()

run_experiment(f"config_files/{args.config}.yml")
