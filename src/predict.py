"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import os
import pickle
from src.config import ROOT_PATH, load_config
import argparse


class MakePredictionPipeline(object):
        """
        Class implementing models prediction. Receives as `input_path` transformed data output of feature_engineering.py,
        laods model from `model_path`, and writes predictions on `output_path`.
        """
        def __init__(self, input_path, output_path, model_path):
            self.input_path = input_path
            self.output_path = output_path
            self.model_path = model_path
                    

        def load_data(self) -> pd.DataFrame:
            """
            Reads test csv that contains the data that will be predicted
            :return pred_data: DataFrame containing the test data. 
            :rtype: pd.DataFrame
            """
            pred_data = pd.read_csv(os.path.join(self.input_path, 'test_final.csv'))

            return pred_data



        def load_model(self) -> None:
            """
            Loads best model. Imports pickle file.
            """    

            #ToDo: change to PRODUCTION_PATH eventually
            config = load_config(os.path.join(ROOT_PATH, "config.yaml"))
            model = config["MODELS_PATH"]

            with open(model, 'rb') as model_file:
                self.model = pickle.load(model_file)           


       # def make_predictions(self, data: DataFrame) -> pd.DataFrame:
            """
            COMPLETAR DOCSTRING
            """
    
       #     new_data = self.model.predict(data)

        #   return new_data

        #def write_predictions(self, predicted_data: DataFrame) -> None:
            """
            COMPLETAR DOCSTRING
            """

         #   return None 

        def run(self):

            data = self.load_data()
            self.load_model()
          #  df_preds = self.make_predictions(data)
          #  self.write_predictions(df_preds)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--models-path", type=str, help="Path to save the trained model in pkl format.")

    args = parser.parse_args()
    MakePredictionPipeline(input_path = args.input_path,
                            models_path = args.models_path).run()


if __name__ == "__main__":
    main()






#if __name__ == "__main__":
    
#   spark = Spark()
    
 #   pipeline = MakePredictionPipeline(input_path = 'Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
  #                                    output_path = 'Ruta/Donde/Voy/A/Escribir/Mis/Datos',
   #                                   model_path = 'Ruta/De/Donde/Voy/A/Leer/Mi/Modelo')
  #  pipeline.run() 