"""
train.py

DESCRIPCIÓN: TRAINING DE MODELO DE ML CON ESCRITURA EN FORMATO .PKL
AUTOR:MARIANA TAGLIO
FECHA: 07/08/2022
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle
import argparse
import os


class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Read the training data from a csv file and returns a pandas DataFrame.        
        :return pandas_df: DataFrame containing the training data. 
        :rtype: pd.DataFrame
        """
        df_train = pd.read_csv(os.path.join(self.input_path, 'train_final.csv'))

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
        X = df.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

        # Train model
        model.fit(x_train,y_train)

        # Prediction over validation set
        pred = model.predict(x_val)

        # Calculate mean square error and rr (coefficient of determination)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        print('Métricas del Modelo:')
        print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        print('\nCoeficientes del Modelo:')
        # Print model constant
        print('Intersección: {:.2f}'.format(model.intercept_))

        # Model coefficients
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Coeficiente Estimados'] = model.coef_
        print(coef, '\n')
        
        return df

  
    def model_dump(self, model_trained) -> None:
        """
        Save a trained machine learning model in pkl format.
        
        :param model_trained: The trained machine learning model to be saved.
        :type model_trained: object
        :return: None
        """
   
        #Save the model as model.pkl in the specified directory
        model_file_path = os.path.join(os.path.dirname(self.model_path), 'model.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_trained, f)

        print(f"Model saved successfully in '{self.model_path}', as 'model.pkl'.")

        return None
          

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model in pkl format.")

    args = parser.parse_args()
    ModelTrainingPipeline(input_path = args.input_path,
                            model_path = args.model_path).run()


if __name__ == "__main__":
    main()

