import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats


data_train = pd.read_csv('../data/Train_BigMart.csv')
data_test = pd.read_csv('../data/Test_BigMart.csv')
# Identificando la data de train y de test, para posteriormente unión y separación
data_train['Set'] = 'train'
data_test['Set'] = 'test'

#Combinando los dataset de *entrenamiento y test* para proceder a realizar la exploración, visualización, limpieza de datos, y posterior ingeniería de características y codificación de variables.
data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

#LIMPIEZA

# FEATURES ENGINEERING: para los años del establecimiento
data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']
#Unificando etiquetas para Item_fat__content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
#Limpieza de faltantes en el peso de los productos
productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
for producto in productos:
    moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
    data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda
#limpieza de faltantes en el tamaño de las tiendas
outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
for outlet in outlets:
    data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

    # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'

data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

# FEATURES ENGINEERING: creando categorías para 'Item_Type'
data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
 'Seafood': 'Meats', 'Meat': 'Meats',
 'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
 'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

# FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

print(pd.qcut(data['Item_MRP'], 4,).unique())
data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

#Se utiliza una copia de data para separar los valores codificados en un dataframe distinto.
dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

# Codificación de variables ordinales
dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos

dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])

#PREPARANDO DATA DE ENTRENAMIENTO Y TEST
# Eliminación de variables que no contribuyen a la predicción por ser muy específicas
dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

# División del dataset de train y test
df_train = dataset.loc[data['Set'] == 'train']
df_test = dataset.loc[data['Set'] == 'test']

# Eliminando columnas sin datos
df_train.drop(['Set'], axis=1, inplace=True)
df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

# Guardando los datasets
df_train.to_csv("train_final.csv")
df_test.to_csv("test_final.csv")