import subprocess
import os 

from config import ROOT_PATH, load_config

def main():

    config = load_config(os.path.join(ROOT_PATH, "config.yaml"))
    
    subprocess.run('Python feature_engineering.py --input-path {DATA_PATH} --output-path {TFMD_DATA_PATH}'.format(**config).split())
    subprocess.run('Python predict.py --input-path {TFMD_DATA_PATH}/test_final.csv --models-path {MODELS_PATH} --output-path {PREDS_PATH}'.format(**config).split())

