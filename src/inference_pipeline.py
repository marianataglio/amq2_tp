import subprocess
import argparse


parser = argparse.ArgumentParser(description="Train pipeline script")
parser.add_argument("--input-path", required=True, help="Name of the input file in transformed path")
parser.add_argument("--models-path", required=True, help="Path to model .pkl")
parser.add_argument("--output-path", required=True, help="Path to predictions path")
parser.add_argument("--log-file", required=True, help="Path to log file")

args = parser.parse_args()

config = dict(input_path = args.input_path,
                models_path = args.models_path,
                output_path = args.output_path,
                log_file = args.log_file
               )

predict_command = 'python predict.py --input-path {input_path} --models-path {models_path} --output-path {output_path}'.format(**config)

#.split() 


with open(args.log_file, 'a') as log:
    pred_process = subprocess.Popen(predict_command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for process in [pred_process]:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            log.write(line.decode('utf-8'))
            log.flush()


