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
        #AÑOS DEL ESTABLECIMIENTO
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        #FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        'Seafood': 'Meats', 'Meat': 'Meats',
        'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
        'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})
       
        #NORMALIZACIÓN ITEM_FAT_CONTENT
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        #FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df.loc[df['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

        #FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        #IMPUTACION POR MODA DE FALTANTES DE ITEM_WEIGHT
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        #LIMPIEZA DE FALTANTES EN EL TAMAÑO DE LAS TIENDAS
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        #SEPARACION DE PRECIO EN CUANTILES
        print(pd.qcut(df['Item_MRP'], 4,).unique())
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels = [1, 2, 3, 4])

        #CODIFICACIÓN DE VARIABLES ORDINALES
        df['Outlet_Size'] = df['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
        df = pd.get_dummies(df, columns=['Outlet_Type'], drop_first=True)

        #PREPARANDO DATA DE ENTRENAMIENTO Y TEST
        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        return df

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write tranformed dataframe to the output_path
        :param transformed_dataframe: the tranformed dataframe to be written
        :type transformed_dataframe: pd.Dataframe
        """
 
        transformed_dataframe.to_csv(self.output_path, index=False)
        #df_test.to_csv("test_final.csv")
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data_train(df_transformed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")

    args = parser.parse_args()
    FeatureEngineeringPipeline(input_path = args.input_path,
                               output_path = args.output_path).run()


if __name__ == "__main__":
    main()
    