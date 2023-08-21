import subprocess
import argparse


parser = argparse.ArgumentParser(description="Train pipeline script")
parser.add_argument("--input-path", required=True, help="Name of the input file in transformed path")
parser.add_argument("--models-path", required=True, help="Path to model .pkl")
parser.add_argument("--output-path", required=True, help="Path to predictions path")

args = parser.parse_args()

config = dict(input_path = args.input_path,
                models_path = args.models_path,
                output_path = args.output_path
               )

subprocess.run(
    'Python predict.py --input-path {input_path} --models-path {models_path}' 
                     ' --output-path {output_path}'.format(**config).split()
)