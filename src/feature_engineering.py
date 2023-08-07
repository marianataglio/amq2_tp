"""
feature_engineering.py


DESCRIPCIÓN: TRANSFORMACIÓN DE VARIABLES
AUTOR: MARIANA TAGLIO
FECHA: 20/07/2023
"""

# Imports
import pandas as pd
import os
import argparse


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING  
        Reads train and test data, concatenates both sets to perform feature engineering. 
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        data_train = pd.read_csv(os.path.join(self.input_path, 'Train_BigMart.csv'))
        data_test = pd.read_csv(os.path.join(self.input_path,'Test_BigMart.csv'))
        # Identificando la data de train y de test, para posteriormente unión y separación
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'
        data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

        return data

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering on dataframe: cleaning and transformation of variables
        :param df: The dataframe that will be transformed
        :type df: pd.Dataframe
        :return: The transformed dataframe
        :rtype: pd.Dataframe
    
        """
        # Outlet establishment year
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        # Mode imputation of item_weight
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # Clean nulls in outlet_size
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        # Divide in buckets of price
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels = [1, 2, 3, 4])

        # Ordinal variables encoding
        df['Outlet_Size'] = df['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
        
        # One hot encoding of outlet_type
        df = pd.get_dummies(df, columns=['Outlet_Type'])

        # Prepare data fron train and test
        # Remove variables that are too specific. 
        df = df.drop(columns=['Item_Fat_Content', 'Item_Type', 'Item_Identifier', 'Outlet_Identifier'])

        df_transformed = df.copy()

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write tranformed dataframe to the output_path
        :param transformed_dataframe: the tranformed dataframe to be written
        :type transformed_dataframe: pd.Dataframe
        """
        # Split transformed df into train and test df based on the set column.
        df_train = transformed_dataframe[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe[transformed_dataframe['Set'] == 'test']

        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Set', 'Item_Outlet_Sales'], axis=1, inplace=True)

        train_output_path = os.path.join(self.output_path, 'train_final.csv')
        test_output_path = os.path.join(self.output_path, 'test_final.csv')

        # Write train df to csv
        df_train.to_csv(train_output_path, sep=',', index=False)

        # Write test df to csv
        df_test.to_csv(test_output_path, sep=',', index=False)  
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)
        
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")

    args = parser.parse_args()
    FeatureEngineeringPipeline(input_path = args.input_path,
                               output_path = args.output_path).run()


if __name__ == "__main__":
    main()
    