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
        
        return df

  
    def model_dump(self, model_trained) -> None:
        """
        Save a trained machine learning model in pkl format.
        
        :param model_trained: The trained machine learning model to be saved.
        :type model_trained: object
        :return: None
        """
   
        # Create model directory if it doesn't exist
        os.makedirs(self.models_path, exist_ok=True)

        # Save the model as model.pkl in the specified directory
        model_file_path = os.path.join(self.models_path, 'model.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_trained, f)

        print(f"Model saved successfully in '{model_file_path}.")
         

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--models-path", type=str, help="Path to save the trained model in pkl format.")

    args = parser.parse_args()
    ModelTrainingPipeline(input_path = args.input_path,
                            models_path = args.models_path).run()


if __name__ == "__main__":
    main()

