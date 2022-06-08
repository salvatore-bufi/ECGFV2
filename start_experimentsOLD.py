from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--model', type=str, default='egcfv2')
parser.add_argument('--dataset', type=str, default='yahoo_10core')
args = parser.parse_args()

run_experiment(f"config_files/{args.model}/{args.dataset}.yml")
