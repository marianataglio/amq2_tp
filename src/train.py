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
        Initialize the ModelTrainingPipeline object.
        
        :param input_path: Path to the input data CSV file and pkl.
        :type input_path: str
        :param models_path: Path to save the trained model in pkl format.
        :type models_path: str
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

    def _evaluate_metrics(self, model, X, y, dataset_name):
        """
        Evaluate and print metrics for the trained model.
        
        :param model: The trained machine learning model.
        :type model: sklearn.base.BaseEstimator
        :param X: Features for prediction.
        :type X: pd.DataFrame
        :param y: Target variable.
        :type y: pd.Series
        :param dataset_name: Name of the dataset (e.g., "Train" or "Test").
        :type dataset_name: str
"""
        mse_train = metrics.mean_squared_error(y, model.predict(X))
        r2_train = model.score(X, y)
        print('Métricas del Modelo:')
        print('{}: RMSE: {:.2f} - R2: {:.4f}'.format(dataset_name, mse_train**0.5, r2_train))

    def _print_model_coefficients(self, model, column_names):
        """
        Print the coefficients of the trained model.
        
        :param model: The trained machine learning model.
        :type model: sklearn.base.BaseEstimator
        :param column_names: List of feature column names.
        :type column_names: list
        """

        print('\nCoeficientes del Modelo:')
        print('Intersección: {:.2f}'.format(model.intercept_))

        # Model coefficients
        coef = dict(zip(column_names, model.coef_))
        print(coef, '\n')

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

            X = df.drop(columns='Item_Outlet_Sales') 
            x_train, x_val, y_train, y_val = train_test_split(
                X, df['Item_Outlet_Sales'], 
                test_size=0.3, random_state=seed
            )
            print("data splited")

            model.fit(x_train, y_train)
            print("model fit")

            pred = model.predict(x_val)

            self._evaluate_metrics(model, x_train, y_train, "Train")
            self._evaluate_metrics(model, x_val, y_val, "Test")

            self._print_model_coefficients(model, x_train.columns)
            
            return model

    def run_pipeline(self):
        """
        Run the model training pipeline
        """
        with open(os.path.join(self.input_path, 'pipeline.pkl'), 'rb') as f:
            features_pipe = pickle.load(f)

        df = self.read_data()
        model = self.model_training(df)
  
        # Access to transformation steps of Pipeline object
        tfms = features_pipe.steps
        tfms = list(tfms)

        # Append the trained model to the existing pipeline
        tfms.append(('model', model))  
        full_pipe = Pipeline(tfms)

        # Save the complete pipeline to a pickled file
        with open(self.models_path, "wb") as f:
            pickle.dump(full_pipe, f)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file and pkl")
    parser.add_argument("--models-path", type=str, help="Path to save the trained model in pkl format.")

    args = parser.parse_args()
    pipeline = ModelTrainingPipeline(input_path=args.input_path, models_path=args.models_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
