"""
feature_engineering.py


DESCRIPCIÓN: TRANSFORMACIÓN DE VARIABLES
AUTOR: MARIANA TAGLIO
FECHA: 18/08/2023
"""


import argparse
import os
from sklearn.pipeline import Pipeline
import pickle


import pandas as pd
from transformations import ModeImputation, OutletYearTransformer, NullImputer, OrdinalEncoderTransformer, \
    OneHotEncoder, PriceBucketsTransformer, RemoveColumns

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

        data = pd.read_csv(self.input_path)

        return data

    def run_pipeline(self):
        df = self.read_data()

        price_columns = ['Item_MRP']
        impute_mode_columns = ['Item_Weight']
        outlet_year_columns = ['Outlet_Establishment_Year']
        null_imputer_columns = ['Outlet_Size',]
        null_value = 'Small'
        ordinal_encoder_mapping = {
            'Outlet_Size': {'High': 2, 'Medium': 1, 'Small': 0},
            'Outlet_Location_Type': {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}
        }
        one_hot_columns = ['Outlet_Type']
        columns_to_drop = ['Item_Fat_Content', 'Item_Type', 'Item_Identifier', 'Outlet_Identifier']


 
        pipeline = Pipeline([
            ('price_buckets', PriceBucketsTransformer(columns=price_columns, num_bins=4, labels=[1,2,3,4])),
            ('impute_mode', ModeImputation(columns=impute_mode_columns)),
            ('outlet_year', OutletYearTransformer(columns=outlet_year_columns)),
            ('outlet_size_imputer', NullImputer(value= null_value, columns=null_imputer_columns)),
            ('ordinal_encoder', OrdinalEncoderTransformer(column_mappings=ordinal_encoder_mapping)),
            ('one_hot', OneHotEncoder(columns=one_hot_columns)),
            ('remove_columns', RemoveColumns(columns=columns_to_drop))
        ])

        # Set the mappings for the ordinal encoder
        #ordinal_encoder = pipeline.named_steps['ordinal_encoder']
        #for column, mapping in ordinal_encoder_mapping.items():
        #    ordinal_encoder.set_mapping(column, mapping)

        # Fit and transform the data using the pipeline
        transformed_data = pipeline.fit_transform(df)

        # Save the transformed data to pkl
        with open(os.path.join(self.output_path, "data_transformed.pkl"), "wb") as pkl_file:
            pickle.dump(pipeline, pkl_file)
        
        transformed_df = pd.DataFrame(transformed_data)

        # Drop columns
        #columns_to_drop = ['Item_Fat_Content', 'Item_Type', 'Item_Identifier', 'Outlet_Identifier']
        #transformed_df.drop(columns=columns_to_drop, inplace=True)

        # Save the transformed data to csv
        transformed_df.to_csv(os.path.join(self.output_path, "train_transformed.csv"), index=False)

   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--output-path", type=str, help="Path to the output data CSV file and pkl.")
    args = parser.parse_args()

    pipeline = FeatureEngineeringPipeline(input_path = args.input_path,
                               output_path = args.output_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
    
