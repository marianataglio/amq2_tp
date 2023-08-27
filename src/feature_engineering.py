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
from transformations import YearTransformer, NullImputer, OrdinalEncoderTransformer, \
    OneHotEncoder, PriceBucketsTransformer, RemoveColumns, ModeImputationByGroup


class FeatureEngineeringPipeline(object):
    """
        Initializes the FeatureEngineeringPipeline object.
        
        :param input_path: Path to the input data CSV file.
        :type input_path: str
        :param output_path: Path to the output directory for transformed data CSV file and pkl.
        :type output_path: str
        """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads data to perform feature engineering. 
        :return data: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        data = pd.read_csv(self.input_path)

        return data

    def run_pipeline(self):
        """
        Runs the feature engineering pipeline
        """

        df = self.read_data()

        price_columns = ['Item_MRP']
        impute_columns = 'Item_Weight'
        group_column = 'Item_Identifier'
        outlet_year_columns = ['Outlet_Establishment_Year']
        null_imputer_columns = ['Outlet_Size', ]
        null_value = 'Small'
        ordinal_encoder_mapping = {
            'Outlet_Size': {'High': 2, 'Medium': 1, 'Small': 0},
            'Outlet_Location_Type': {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}
        }
        one_hot_columns = ['Outlet_Type']
        columns_to_drop = ['Item_Fat_Content', 'Item_Type', 'Item_Identifier', 'Outlet_Identifier']

        pipeline = Pipeline([
            ('price_buckets', PriceBucketsTransformer(columns=price_columns, num_bins=4, labels=[1, 2, 3, 4])),
            ('impute_mode', ModeImputationByGroup(impute_column=impute_columns, group_column=group_column)),
            ('outlet_year', YearTransformer(columns=outlet_year_columns)),
            ('outlet_size_imputer', NullImputer(value=null_value, columns=null_imputer_columns)),
            ('ordinal_encoder', OrdinalEncoderTransformer(column_mappings=ordinal_encoder_mapping)),
            ('one_hot', OneHotEncoder(columns=one_hot_columns)),
            ('remove_columns', RemoveColumns(columns=columns_to_drop))
        ])

        # Fit and transform the data using the pipeline
        transformed_data = pipeline.fit_transform(df)
        print("data transformed")

        # Save the transformed data to pkl
        pkl_path = os.path.join(self.output_path, "pipeline.pkl")
        print(f"Saving pipeline to {pkl_path}")

        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(pipeline, pkl_file)

        transformed_df = pd.DataFrame(transformed_data)

        # Save the transformed data to csv
        csv_path = os.path.join(self.output_path, "train_transformed.csv")
        print(f"Saving transformed data to: {csv_path}")
        transformed_df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("--output-path", type=str, help="Path to the output data CSV file and pkl.")
    args = parser.parse_args()

    pipeline = FeatureEngineeringPipeline(input_path=args.input_path,
                                          output_path=args.output_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
