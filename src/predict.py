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
from config import ROOT_PATH, load_config
import argparse


class MakePredictionPipeline(object):
        """
        Class implementing models prediction. Receives as `input_path` transformed data output of feature_engineering.py,
        laods model from `model_path`, and writes predictions on `output_path`.
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
            data_to_pred = pd.read_csv(os.path.join(self.input_path))
            return data_to_pred


        def load_model(self) -> None:
            """
            Loads best model. Imports pickle file.
            """    
             # Print the model path for debugging
            print("Model path:", self.models_path)
            #ToDo: change to PRODUCTION_PATH eventually

            with open(os.path.join(self.models_path, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)           
            print("Pipeline loaded:", self.model) 


        def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            COMPLETAR DOCSTRING
            """
            #Apply transformations 
            model_pipeline = self.model
            predicted_data = model_pipeline.named_steps['model'].predict(data)
            return predicted_data

           # predicted_data = self.model.predict(data)
           # return predicted_data
       

        def write_predictions(self, predicted_data: pd.DataFrame) -> None:
            """
            COMPLETAR DOCSTRING
            """ 
            output_df = pd.DataFrame({'Predicted_Item_Outlet_Sales': predicted_data})
            output_df.to_csv(self.output_path, index=False)
         #   return None 

        def run(self):

            data = self.load_data()
            self.load_model()
            df_preds = self.make_predictions(data)
            self.write_predictions(df_preds)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--models-path", type=str, help="Path to load the trained model in pkl format.")
    parser.add_argument("--output-path", type=str, help="Path to save predictions")

    args = parser.parse_args()
    MakePredictionPipeline(input_path = args.input_path, models_path = args.models_path, output_path = args.output_path).run()


if __name__ == "__main__":
    main()






#if __name__ == "__main__":
    
#   spark = Spark()
    
 #   pipeline = MakePredictionPipeline(input_path = 'Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
  #                                    output_path = 'Ruta/Donde/Voy/A/Escribir/Mis/Datos',
   #                                   model_path = 'Ruta/De/Donde/Voy/A/Leer/Mi/Modelo')
  #  pipeline.run() 