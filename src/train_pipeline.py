import subprocess
import argparse

parser = argparse.ArgumentParser(description="Train pipeline script")
parser.add_argument("--input-file", required=True, help="Name of the input file in DATA_PATH")
parser.add_argument("--tfmd-dir", required=True, help="Path to transformed csv and pkl")
parser.add_argument("--model-file", required=True, help="Path to model .pkl")
parser.add_argument("--log-file", required=True, help= "Path to log file")

args = parser.parse_args()

config = dict(input_file = args.input_file,
                tfmd_dir = args.tfmd_dir,
                model_file = args.model_file,
                log_file = args.log_file)



fe_command = (
    'python feature_engineering.py --input-path {input_file} --output-path {tfmd_dir}'
    .format(**config)
)

train_command = (
    'python train.py --input-path {tfmd_dir} --models-path {model_file}'
    .format(**config)
)


with open(args.log_file, 'a') as log:
    fe_process = subprocess.Popen(fe_command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    train_process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for process in [fe_process, train_process]:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            log.write(line.decode('utf-8'))
            log.flush()
    
    fe_process.wait()
    train_process.wait()
