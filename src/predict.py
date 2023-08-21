"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import pickle
import argparse


class MakePredictionPipeline(object):
        """
        Class implementing models prediction. Receives as `input_path` transformed data output of feature_engineering.py,
        loads model from `model_path`, and writes predictions on `output_path`.
        """
        def __init__(self, input_path, output_path, models_path):
            self.input_path = input_path
            self.output_path = output_path
            self.models_path = models_path
                    

        def load_data(self) -> pd.DataFrame:
            """
            Reads test csv that contains the data that will be predicted
            :return pred_data: DataFrame containing the test data. 
            :rtype: pd.DataFrame
            """
            return pd.read_csv(self.input_path)

        def load_model(self) -> None:
            """
            Loads best model. Imports pickle file.
            """    
             # Print the model path for debugging
            print("Model path:", self.models_path)
            #ToDo: change to PRODUCTION_PATH eventually

            with open(self.models_path, 'rb') as f:
                self.model = pickle.load(f)           
            print("Pipeline loaded:", self.model) 

        def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            COMPLETAR DOCSTRING
            """
            #predict 
            model_pipeline = self.model
            predicted_data = model_pipeline.predict(data)
            data["Prediction"] = predicted_data

        def write_predictions(self, data: pd.DataFrame) -> None:
            """
            COMPLETAR DOCSTRING
            """ 
            data.to_csv(self.output_path, index=False)

        def run(self):

            data = self.load_data()
            self.load_model()
            self.make_predictions(data)
            self.write_predictions(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--models-path", type=str, help="Path to load the trained model in pkl format.")
    parser.add_argument("--output-path", type=str, help="Path to save predictions")

    args = parser.parse_args()
    MakePredictionPipeline(input_path = args.input_path, models_path = args.models_path, output_path = args.output_path).run()


if __name__ == "__main__":
    main()