"""
train.py

DESCRIPCIÓN: TRAINING DE MODELO DE ML CON ESCRITURA EN FORMATO .PKL
AUTOR:MARIANA TAGLIO
FECHA: 07/08/2022
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModelTrainingPipeline(object):
    """
    Class that handles the whole training pipeline

    """
    def __init__(self, input_path, models_path):
        """
        :input_path str 
        """
        self.input_path = input_path
        self.models_path = models_path

    def read_data(self) -> pd.DataFrame:
        """
        Read the training data from a csv file and returns a pandas DataFrame.        
        :return df_train: DataFrame containing the training data. 
        :rtype: pd.DataFrame
        """
        df_train = pd.read_csv(os.path.join(self.input_path, 'train_transformed.csv'))

        return df_train

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Train a linear regression model using the provided DataFrame.

            :param df: The DataFrame containing the training data.
            :type df: pd.DataFrame
            :return: The DataFrame with training data (unchanged).
            :rtype: pd.DataFrame
            """
            seed = 28
            model = LinearRegression()

            # Split into training and validation sets
            X = df.drop(columns='Item_Outlet_Sales') 
            x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

            # Train model
            model.fit(x_train,y_train)

            # Prediction over validation set
            pred = model.predict(x_val)

            # Calculate mean square error and rr (coefficient of determination)
            mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
            r2_train = model.score(x_train, y_train)
            print('Métricas del Modelo:')
            print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, r2_train))

            mse_val = metrics.mean_squared_error(y_val, pred)
            r2_val = model.score(x_val, y_val)
            print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, r2_val))

            print('\nCoeficientes del Modelo:')
            # Print model constant
            print('Intersección: {:.2f}'.format(model.intercept_))

            # Model coefficients
            coef = pd.DataFrame(x_train.columns, columns=['features'])
            coef['Coeficiente Estimados'] = model.coef_
            print(coef, '\n')
            
            return model

    def run_pipeline(self):

        # Load the pre-trained pipeline from a pickled file
        with open(os.path.join(self.input_path, 'data_transformed.pkl'), 'rb') as f:
            features_pipe = pickle.load(f)

        #Train model
        df = self.read_data()
        model = self.model_training(df)
  
        #Access to transformation steps of Pipeline object
        tfms = features_pipe.named_steps.items()
        tfms = list(tfms)

        # Append the trained model to the existing pipeline
        tfms.append(('model', model))  
        full_pipe = Pipeline(tfms)

        # Save the complete pipeline to a pickled file
        with open(os.path.join(self.models_path, "full_pipeline.pkl"), "wb") as f:
            pickle.dump(full_pipe, f)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file and pkl")
    parser.add_argument("--models-path", type=str, help="Path to save the trained model in pkl format.")

    args = parser.parse_args()
    pipeline = ModelTrainingPipeline(input_path = args.input_path,
                            models_path = args.models_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
