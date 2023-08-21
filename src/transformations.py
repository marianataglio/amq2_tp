
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
import warnings


class PriceBucketsTransformer(BaseEstimator, TransformerMixin):
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
        
 
# Mode imputation
class ModeImputation(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.most_frequent_values = X[self.columns].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        missing_rows = X[self.columns].isnull().any(axis=1)
        X.loc[missing_rows, self.columns] =  self.most_frequent_values.values
        return X


#Outlet year transformation (it will receive the outlet establishment year and it will convert it to 2020-year)
#si quisiera mejorarlo tendría que usar datetime.now().year pero no lo voy a hacer ahora porque
#necesito asegurarme de poder reproducir lo que hizo el DS

class OutletYearTransformer(BaseEstimator, TransformerMixin):
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

#Outlet size imputer: converts outlet size null to small based on the associated outlet_identifier

class NullImputer(BaseEstimator, TransformerMixin):
    def __init__(self, value= 'Small', columns=None):
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
    def __init__(self, column_mappings):
        self.column_mappings = deepcopy(column_mappings)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column, mapping in self.column_mappings.items():
            if column in X.columns:
                unique_values = set(X[column].unique())
                mapping_keys = set(mapping.keys())
                if unique_values.issuperset(mapping_keys) and unique_values != mapping_keys:
                    diff = unique_values - mapping_keys               
                    raise RuntimeError(f"{diff} values not found in mapping for column {column}")       
                X[column] = X[column].replace(mapping)
            else:
                warnings.warn(f"Column {column} not found in dataset. Continuing")
        return X

# Se puede mejorar usando el onehot encdoing de sklearn, pero quiero asegurarme de tener los mismos resultados que el DS

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_one_hot =pd.get_dummies(X, columns = self.columns, dtype=int)
        return X_one_hot

class RemoveColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        columns_set = set(self.columns)
        found_columns = set(X.columns)
        intersection = columns_set.intersection(found_columns)
        diff = columns_set - found_columns

        #Warning: columns not found in dataset
        if len(diff) > 0:
            warnings.warn("Some columns were not found ({diff}) while dropping. Continuing")
        X.drop(columns=intersection, inplace=True)

        return X