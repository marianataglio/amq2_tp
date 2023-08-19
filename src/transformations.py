
import argparse
import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pickle


#Convert price into quantiles
class PriceBucketsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = pd.qcut(X_transformed[column], 4, labels=[1, 2, 3, 4])
        return X_transformed


# Mode imputation

class ModeImputation(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.most_frequent_values = X[self.columns].mode().iloc[0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        missing_rows = X_copy[self.columns].isnull().any(axis=1)
        X_copy.loc[missing_rows, self.columns] =  self.most_frequent_values.values
        return X_copy


#Outlet year transformation (it will receive the outlet establishment year and it will convert it to 2020-year)
#si quisiera mejorarlo tendría que usar datetime.now().year pero no lo voy a hacer ahora porque
#necesito asegurarme de poder reproducir lo que hizo el DS

class OutletYearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = 2020 - X_transformed[column]
        return X_transformed


#Outlet size imputer: converts outlet size null to small based on the associated outlet_identifier

class OutletSizeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.default_outlet_size = 'Small'
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column].fillna(self.default_outlet_size, inplace=True)
        return X_transformed

        
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_mappings = {}
    
    def set_mapping(self, column, mapping):
        self.column_mappings[column] = mapping
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column, mapping in self.column_mappings.items():
            if column in X_transformed.columns:
                X_transformed[column] = X_transformed[column].replace(mapping)
        return X_transformed

# Se puede mejorar usando el onehot encdoing de sklearn, pero quiero asegurarme de tener los mismos resultados que el DS

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_one_hot =pd.get_dummies(X, columns = self.columns, dtype=int)
        return X_one_hot

