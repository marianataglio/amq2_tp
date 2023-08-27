
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
import warnings


class PriceBucketsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies price bucketing to specified columns
    """
    def __init__(self, columns=None, num_bins=4, labels=[1, 2, 3, 4]):
        self.columns = columns
        self.num_bins = num_bins
        self.labels = labels
        self.bins = {}

    def fit(self, X, y=None):
        for column in self.columns:
            _, self.bins[column] = pd.qcut(X[column], self.num_bins, duplicates='drop', retbins=True)
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = pd.cut(X[column], bins=self.bins[column], labels=self.labels, include_lowest=True)
        return X
    

class ModeImputationByGroup(BaseEstimator, TransformerMixin):
    """
    A transformer that performs mode imputation within groups
    """
    def __init__(self, impute_column, group_column):
        self.impute_column = impute_column
        self.group_column = group_column
        self.group_modes_ = None
        self.mean = None

    def fit(self, X, y=None):
        # Dict to store modes for each group 
        self.group_modes_ = {}

        # Calculate the mean of the impute_column
        self.mean = X[self.impute_column].mean()

        # Find unique values in the group_column where the impute column in null
        productos = list(X[X[self.impute_column].isnull()][self.group_column].unique())

        # Iterate through each unique value in the group_column
        for producto in productos:
            column = (X[X[self.group_column] == producto][[self.impute_column]]).dropna()

            # Check if there are non-null values in the column
            if len(column) > 0:
                # Calculate the mode and store it for the current group
                self.group_modes_[producto] = column.mode().iloc[0,0]
        return self

    def transform(self, X):
        # Find unique values in the group_column where impute_column is null
        productos = list(X[X[self.impute_column].isnull()][self.group_column].unique())

        # Iterate through each unique value in the group_column
        for producto in productos:
            # Check if the mode for the current group is available
            if producto in self.group_modes_:
                # Impute missing values with the mode of the current group
                value = self.group_modes_[producto]
            else:
                # Impute with mean if product id hasn't been seen during train
                value = self.mean
            # Update the missing values in the impute_column with the calculated value
            X.loc[X[self.group_column] == producto, self.impute_column] = value
        return X
     

class YearTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that converts a specific year to number of years from the current year
    """
    def __init__(self, current_year=2020, columns=None):
        self.columns = columns
        self.current_year = current_year
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = self.current_year - X[column]
        return X

class NullImputer(BaseEstimator, TransformerMixin):
    """
    A transformer that imputes null values with a specified values.
    """
    def __init__(self, value='Small', columns=None):
        self.columns = columns
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column].fillna(self.value, inplace=True)
        return X

        
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs ordinal encoding using specified mappings.
    """

    def __init__(self, column_mappings):
        self.column_mappings = deepcopy(column_mappings)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column, mapping in self.column_mappings.items():
            # Check if the current column is present in the input dataframe
            if column in X.columns:
                # Get the unique values present in the column
                unique_values = set(X[column].unique())

                # Get the keys (values to be encoded) from the mapping
                mapping_keys = set(mapping.keys())

                # Check if all mapping keys are present in the column's unique values,
                # and if there are some unique values that are not present in the mapping
                if unique_values.issuperset(mapping_keys) and unique_values != mapping_keys:

                    # Calculate the set difference to find values not found in the mapping
                    diff = unique_values - mapping_keys          
                    # Raise an error indicating which values are missing in the mapping for this column
                    raise RuntimeError(f"{diff} values not found in mapping for column {column}")       
                X[column] = X[column].replace(mapping)
            else:
                # If the current column is not found in the dataset, issue a warning
                warnings.warn(f"Column {column} not found in dataset. Continuing")
        return X

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one hot encoding encoding on specified columns.
    """
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_one_hot = pd.get_dummies(X, columns=self.columns, dtype=int)
        return X_one_hot

class RemoveColumns(BaseEstimator, TransformerMixin):
    """
    A transformer that removes specified columns.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        columns_set = set(self.columns)
        found_columns = set(X.columns)
        intersection = columns_set.intersection(found_columns)
        diff = columns_set - found_columns

        # If the current column is not found in the dataset, issue a warning
        if len(diff) > 0:
            warnings.warn("Some columns were not found ({diff}) while dropping. Continuing")
        X.drop(columns=intersection, inplace=True)

        return X
