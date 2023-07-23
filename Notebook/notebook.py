# %% [markdown]
# # Problema de prediccion de ventas

# %% [markdown]
# ## Planteamiento del problema y Objetivo:

# %% [markdown]
# El objetivo es construir un modelo de regresión simple para predecir las **ventas por producto de una tienda en particular**, que forma parte de una cadena de tiendas, y descubrir cuáles son los **principales factores que influencian dicha predicción**.

# %% [markdown]
# ### Importando Librerías

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats

# %% [markdown]
# ### Lectura de los datasets

# %%
data_train = pd.read_csv('../data/Train_BigMart.csv')

data_test = pd.read_csv('../data/Test_BigMart.csv')

# Identificando la data de train y de test, para posteriormente unión y separación
data_train['Set'] = 'train'
data_test['Set'] = 'test'

# %% [markdown]
# Combinando los dataset de *entrenamiento y test* para proceder a realizar la exploración, visualización, limpieza de datos, y posterior ingeniería de características y codificación de variables.

# %%
data = pd.concat([data_train, data_test], ignore_index=True, sort=False)
data.head(20)

# %% [markdown]
# ## EDA

# %%
print('Variables:', list(data.columns))

# %% [markdown]
# ###  Resumen de los datasets

# %%
print('Número de registros de train:', data_train.shape[0], '  -   Número de variables:', data_train.shape[1])
print('Número de registros de test:', data_test.shape[0], '  -   Número de variables:', data_test.shape[1])
print('Número de registros totales:', data.shape[0], '  -   Número de variables:', data.shape[1])

# %% [markdown]
# Visión general de las variables en cada dataset:

# %%
print('Dataset de entrenamiento:\n')
data_train.info()
print('\nDataset de test:\n')
data_test.info()
print('\nDataset de total:\n')
data.info()
#data_train.info(), data_test.info(), data.info()

# %% [markdown]
# En función del tipo de variables observado, las variables de tipo object corresponde a variables categóricas que deberán ser codificadas. También se observan algunos datos faltantes que deberán ser imputados.
# 
# Se tiene una columna más en los datasets de train y total, correspondiente al Target (Item_Outlet_Sales).

# %% [markdown]
# ### Variables:

# %% [markdown]
# - Item_Identifier: nombre o identificador del producto 
# - Item_Weight: peso del producto (en gramos)
# - Item_Fat_Content: clasificación del producto en términos de grasas contenidas en él. 
# - Item_Visibility: scoring de visibilidad del producto: medida que hace referencia al conocimiento del producto en el consumidor. ¿Qué tan fácil puede ser encontrado el producto? 
# - Item_Type: tipo de producto 
# - Item_MRP: máximum retailed price. Precio calculado por el fabricante que indica el precio más alto que se puede cobrar por el producto. 
# - Outlet_Identifier: identificador de la tienda 
# - Outlet_Establishment_Year: año de lanzamiento de la tienda 
# - Outlet_Size: tamaño de la tienda 
# - Outlet_Location_Type: clasificación de las tiendas según ubicación 
# - Outlet_Type: tipo de tienda 
# - Item_Outlet_Sales: ventas del producto en cada observacion

# %% [markdown]
# ## Planteamiento de Hipótesis:
# Respecto a las variables que se disponen en el dataset y de acuerdo al objetivo propuesto, se plantean algunas hipótesis:
# - El peso del producto no debería influir en los niveles de venta de la tienda.
# - El contenido de grasas de los productos puede ser significativo pra el nivel de venta (Los productos con mayor contenido de grasa quiezás se compran menos).
# - La visibilidad de un producto incide en el nivel de venta de la tienda (generalmente los productos más costosos se exhiben en sitios de fácil visualización para el cliente).
# - El tipo de producto puede influir en el nivel de ventas (existe productos de mayor y menor rotación, pero también de mayor y menor precio).
# - El precio de un producto es un factor que está directamente asociado con el nivel de ventas.
# - El año de lanzamiento de la tienda, da información del tiempo de vida que puede tener la tienda; esto podría influir en el nivel del conocimiento que tiene el cliente de la existencia de la tienda, y por ende de su nivel de ventas.
# - A mayor tamaño de la tienda, mayor nivel de ventas. Las personas le suelen gustar los lugares amplios para ir de compras.
# - La ubicación de la tienda es un factor preponderante en el acceso al cliente y por ende en el nivel de ventas.

# %% [markdown]
# ### Análisis univariado

# %% [markdown]
# ### Resumen estadístico de variables cuantitativas o numéricas:
# Obtener más información de los datos a través de el comportamiento y distribución de los mismos.

# %%
data.describe()

# %% [markdown]
# No se observan valores llamativos en los descriptores estadísticos de cada distribución.
# 
# Refleja valores perdidos en la variable "Item_Weight" (la diferencia de valores en la variable "Item_Outlet_Sales" corresponde a los valores de TARGET en el train dataset)
# 
# **La variable "Outlet_Establishment_Year" será tomada como vida del establecimiento en años, la cual puede dar una información más valiosa.**

# %% [markdown]
# ### Visualizando las variables numéricas:

# %%
# Visualización de las caraterísticas númericas de entrada
data.hist(column=['Item_Weight', 'Item_Visibility',	'Item_MRP', 'Item_Outlet_Sales'], figsize=(26,4), bins=30, layout=(1,4))
plt.show()

# %% [markdown]
# TARGET: Las ventas de la tiendas (Item_Outlet_Sales) presentan una distribución con sesgo positivo, es decir, sus valores se concentran más en los niveles de ventas inferiores.
# 
# Los pesos de los productos (Item_Weight) presentan una distribución clara, no se encuentra concentración de frecuencias en valores específicos.
# 
# La visibilidad de los productos (Item_Visibility) también presenta una distribución sesgada positivamente, se observa mayor concentración en valores inferiores.
# 
# El precio máximo por producto (Item_MRP) presenta una distribución multimodal, de aproximadamente 4 niveles de precios distintos.
# 
# *Las variables sesgadas se les tratará para eliminar dicho sesgo.*
# 
# - Por ahora, se realizará el cálculo de años de vida de la tienda en base al año de establecimiento y el año actual (se asume que es data del actual año 2019):

# %% [markdown]
# #### FEATURES ENGINEERING: para los años del establecimiento

# %%
data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']

# %% [markdown]
# ## Definiendo las variables categóricas

# %% [markdown]
# ### Resumen estadístico de variables categóricas:

# %%
data.describe(include = ['object', 'category'])

# %% [markdown]
# - Item_Identifier posee muchos valores únicos que no se podrán analizar de esta manera tan dispersa, se puede tratar de agrupar según alguna patrón de la codificación.
# - Item_Type también posee un número de características que se podrían agrupar para evitar trabajar con 16 valores; de ser conveniente para la predicción.
# - Las demás variables tienen número de categorías finitas convenientes para el análisis.
# - Se tienen valores faltantes en la variable Outlet_Size que habrá que trabajar.
# 
# Seguido se hace una exploración más detallada:

# %% [markdown]
# ### Conociendo las variables categóricas:

# %%
categoricals = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for cat in categoricals:
    print(cat, ':', set(data[cat]))

# %% [markdown]
# Del análisis se observa:
# - Para "Item_Fat_Content" diferentes etiquetas para la misma categoría. **Acción**: unificar etiquetas.
# - Se considera reagrupar algunas categorías de "Item_Type".

# %% [markdown]
# #### LIMPIEZA: Unificando etiquetas para 'Item_Fat_Content'

# %%
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

# %% [markdown]
# Verificamos la unificación de etiquetas:

# %%
set(data['Item_Fat_Content'])

# %% [markdown]
# ### Miramos el comportamiento de las frecuencias de las variables categóricas:

# %%
for aux in ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    print('\n', aux, ':\n', data[aux].value_counts())

# %% [markdown]
# ### Visualizando la distribucón de frecuencias de las variables categóricas:

# %%
for var_cat in ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Establishment_Year']:
    ancho_bar = len(data[var_cat].unique())
    plt.figure(figsize=(ancho_bar*1.8,4))
    values = data[var_cat].dropna().sum()
    ax = sns.countplot(x= var_cat, data=data, palette='Set2')
    for p in ax.patches:
        ax.annotate('{:.0f} ({:.1f}%)'.format(p.get_height(), p.get_height()/len(data)*100), (p.get_x()+0.1, p.get_height()+30))
    plt.title('Distribución de Frecuencias de ' + var_cat)
    plt.show()

# %% [markdown]
# - El mayor porcentaje de producto corresponde a los bajos en grasas (aprox 65%)
# - Los productos con mayor registros son los vegetales-frutas y los snacks, seguidos de los productos del hogar, enlatados, lácteos, congelados y horneados. 
# - Las tiendas con menores registros son la OUT10 y OUT19, el resto de las tiendas tienen un número de registros similar.
# - Se tienen mayores registros en la tiendas pequeñas y medianas.
# - El mayor número de registros de ventas lo presentan las tiendas con locación Tier 3 y las tiendas de tipo Supermarket Type1.

# %% [markdown]
# #### Porcentaje de valores perdidos

# %%
print('El porcentaje de valores perdidos de las variables: \n')
for var in data.columns:
    num_nan = data[var].isnull().sum()
    print('{}: \t\t{} ({:,.2f}%)'.format(var, num_nan, num_nan*100/len(data)))

# %% [markdown]
# Se tiene 17,17% de valores perdidos en la variable de pesos del producto, lo cual se puede solucionar asignando el peso de un producto similar o desde otro registro del mismo producto. De similar manera se puede realizar con los valores faltantes (28,27%) de la variable Tamaño del outlet.

# %% [markdown]
# Parte del dataset con valores perdidos en la variable 'Item_Weight':

# %%
data[data['Item_Weight'].isnull()].sort_values('Item_Identifier').head()

# %%
print(list(data[data['Item_Weight'].isnull()]['Outlet_Identifier'].unique()))

# %% [markdown]
# Los valores faltantes de pesos de los productos corresponden a las tiendas cuyo código son 'OUT027' y 'OUT019'

# %%
print(len(list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())))

# %% [markdown]
# Se tienen 1559 productos de los 2439 registros con valores perdidos en la variable 'Item_Weight'

# %% [markdown]
# Ahora se procede a rellenar los faltantes en los registros de pesos, basado en el valor modal del peso del producto. (Imputación de casos similares)

# %% [markdown]
# #### LIMPIEZA: de faltantes en el peso de los productos

# %%
productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
for producto in productos:
    moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
    data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda

# %% [markdown]
# Se verifica que no existan valores nulos para la variable peso del producto.

# %%
print('El porcentaje de valores perdidos de la variable "Item_Weight" es de:', data['Item_Weight'].isnull().sum()/len(data)*100)

# %% [markdown]
# Se procede a revisar los faltantes de la variable tamaño de la tienda.

# %%
data[data['Outlet_Size'].isnull()].sort_values('Item_Identifier').tail(10)

# %%
print(list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique()))

# %% [markdown]
# Los valores faltantes de tamaño de la tienda corresponden a las tiendas cuyo código son 'OUT010', 'OUT045' y 'OUT017'

# %% [markdown]
# Se procede primero a verificar qué valores de tamaño registran estas tiendas.

# %%
outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())

# %%
categoria = data[data['Outlet_Identifier'] == 'OUT010']['Outlet_Size'].unique()
print('OUT010', categoria)

categoria = data[data['Outlet_Identifier'] == 'OUT045']['Outlet_Size'].unique()
print('OUT045', categoria)

categoria = data[data['Outlet_Identifier'] == 'OUT017']['Outlet_Size'].unique()
print('OUT017', categoria)

# %% [markdown]
# Se observa que estas 3 tiendas no tienen registros del tamaño de su tienda. Para dar solución a esto se buscará algún tipo de asociación de otra variable con el tamaño, para realizar la estimación de la categoría.

# %% [markdown]
# ### Análisis Bi-variado:
# Variables Categóricas vs Categóricas:

# %%
sns.catplot(x="Outlet_Size", hue='Outlet_Type', data=data, kind="count", height=3, aspect=2)
plt.title('Outlet Size vs Outlet_Type por Outlet Identifier')
plt.show()

# %% [markdown]
# - La mayoría de los "Supermarket Type 1" son de tamaño "Small".
# - Las tiendas "Grocery Store" son de tamaño "Small".
# - Las tiendas "Supermarket Type 2" y "Supermarket Type 3" son de tamaño "Medium".

# %% [markdown]
# - Outlet_Size vs Outlet_Type

# %%
plt.figure(figsize=(10,6))
sns.heatmap(pd.crosstab(data['Outlet_Size'], data['Outlet_Type'], margins=False, normalize=False), annot=True, square=False, fmt='', cbar_kws={"orientation": "horizontal"}, linewidths=0.5)
plt.show()

# %% [markdown]
# Se observa que no existe una relación entre el tipo de tienda y el tamaño de la misma. 
# - Item_Type vs Outlet_Type

# %%
plt.figure(figsize=(10,12))
sns.heatmap(pd.crosstab(data['Item_Type'], data['Outlet_Type'], normalize=False), annot=True, square=False, fmt='', cbar_kws={"orientation": "horizontal"}, linewidths=0.5)
plt.show()

# %% [markdown]
# El Supermarket Type 2 y 3 presentan distribución similar respecto de los tipos de productos, al igual que en el tamaño de la tienda.
# Vemos:
# - Outlet_Location_Type vs Outlet_Type

# %%
import statsmodels.api as sm
tab = pd.crosstab(data['Outlet_Location_Type'], data['Outlet_Type'], margins=False, normalize=False)
plt.figure(figsize=(10,6))
sns.heatmap(tab, annot=True, square=False, fmt='', cbar_kws={"orientation": "horizontal"}, linewidths=0.5)
plt.show()

# %% [markdown]
# - La mayor cantidad de registros son de la tienda "Supermarket Type 1" y de tamaño "Small"; en primer lugar de la ubicación "Tier 2" y en segundo de  la ubicación "Tier 1".

# %% [markdown]
# Veamos el tamaño de la tienda con respecto al nivel de ventas.

# %% [markdown]
# ### Análisis Bi-variado:
# Variables Categóricas vs Continuas:
# - Veamos por un momento el tipo de tienda respecto a las ventas:

# %%
plt.figure(figsize=(10,4))
sns.violinplot(x=data['Outlet_Type'], y=data["Item_Outlet_Sales"])
plt.show()

# H0: las medias son significativamente iguales entre los grupos (Se utiliza el test de Kruskal-Wallis por tratarse de una variable que no tiene una distribución normal)
print('\n', stats.kruskal(list(data.dropna().loc[data['Outlet_Type']== 'Supermarket Type1', 'Item_Outlet_Sales']), 
            list(data.dropna().loc[data['Outlet_Type']== 'Supermarket Type2', 'Item_Outlet_Sales']),
            list(data.dropna().loc[data['Outlet_Type']== 'Supermarket Type3', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Outlet_Type']== 'Grocery Store', 'Item_Outlet_Sales'])))

# %% [markdown]
# - Se evidencia diferencias significativas en los niveles de ventas por tipo de tienda.
# - La distribución de frecuencia de las variables estudiadas arriba son similares para los tipos de tiendas "Supermarket Type 2" y "Supermarket Type 3"; sin embargo no lo es así el comportamiento de las ventas. Se dejarán estas categorias separadas como están originalmente.

# %%
sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=data)
plt.show()

med=data.dropna().loc[data['Outlet_Size']=='Medium', 'Item_Outlet_Sales']
hig=data.dropna().loc[data['Outlet_Size']=='High', 'Item_Outlet_Sales']
sma=data.dropna().loc[data['Outlet_Size']=='Small', 'Item_Outlet_Sales']

sns.distplot(sma, kde=True, hist=False, label='Small'), sns.distplot(med, kde=True, hist=False, label='Medium'), sns.distplot(hig, kde=True, hist=False, label='High')
plt.show()

# Cálculo de promedios de ventas de cada tamaño de tienda
print('\nVentas promedios (Small):', sma.mean())
print('Ventas promedios (Medium):', med.mean())
print('Ventas promedios (High):', hig.mean())

print('\n', stats.kruskal(list(med), list(hig), list(sma)))  # H0: las medias son significativamente iguales entre los grupos

# %% [markdown]
# Mediante la prueba de Kruskal-Wallis se evidencia diferencias significativas en los niveles de venta para los distintos tamaños de tiendas.
# 
# Se somete a prueba las diferencias estadísticas entre el tamaño de tienda Small y High, para descartar similitud en sus ventas:

# %%
stats.mannwhitneyu(list(hig), list(med))  # H0: las medias son significativamente iguales para ambos grupos

# %% [markdown]
# Se evidencia diferencias significativas entre las ventas promedios de ambos tamaños de tiendas (Medium y High).
# 
# Seguidamente se visualiza el comportamiento de las ventas de las tiendas que presentan VALORES PERDIDOS en el tamaño de tienda (Outlet_Size):

# %%
data_aux = data[data['Outlet_Size'].isnull()]
plt.figure(figsize=(10,4))
sns.boxplot(x="Outlet_Identifier", y="Item_Outlet_Sales", data=data_aux)
plt.show()

# %% [markdown]
# Los valores de ventas en la tienda OUT10 son muy pequeños en comparación a las tiendas OUT17 y OUT45.
# 
# Graficando los diagramas box-plot de los niveles de ventas de las tiendas según tamaño (Oulet_Size) vs tipo de tienda (Outlet_Type):

# %%
plt.figure(figsize=(15,4))
sns.boxplot(x="Outlet_Identifier", y="Item_Outlet_Sales", hue='Outlet_Size', data=data)
plt.show()

# %% [markdown]
# No se muestra algún patrón que se deba destacar.
# 
# Graficando diagramas box-plot de los niveles de ventas de las tiendas según el tipo de tienda (Outlet_Type):

# %%
plt.figure(figsize=(15,6))
sns.boxplot(x="Outlet_Identifier", y="Item_Outlet_Sales", hue='Outlet_Type', data=data)
plt.show()

# %% [markdown]
# Se observa que la tienda OUT10 tiene un comportamiento similar en el nivel de ventas, que las tiendas OUT17 y OUT45 tienen coportamientos similares en sus ventas a las tiendas OUT13 y OUT46 respectivamente.

# %% [markdown]
# Se decide asignar a todos los valores perdidos del tamaño de las tiendas, la categoria "Small".
# 
# Tomando en consideración lo siguiente:
# - El OUT10 es una tienda de tipo "Grocery Store" (lo que implica ser una tienda pequeña) y además tiene unas ventas similares al OUT19.
# - El OUT17 es una tienda de tipo "Supermarket Type 1" (la mayoría de las tiendas "Supermarket Type 1" son de tamaño "Small").
# - El OUT45 es una tienda de tipo "Supermarket Type 1" (la mayoría de las tiendas "Supermarket Type 1" son de tamaño "Small").

# %% [markdown]
# #### LIMPIEZA: de faltantes en el tamaño de las tiendas

# %%
#data.loc[data['Outlet_Identifier'] == 'OUT10', 'Outlet_Size'] =  'Small'
#data.loc[data['Outlet_Identifier'] == 'OUT17', 'Outlet_Size'] =  'Small'
#data.loc[data['Outlet_Identifier'] == 'OUT45', 'Outlet_Size'] =  'Small'

# %%
for outlet in outlets:
    data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

# %% [markdown]
# Se verifica que no existan valores nulos para la variable peso del producto.

# %%
print('El porcentaje de valores perdidos de la variable "Outlet_Size" es de:', data['Outlet_Size'].isnull().sum()/len(data)*100)

# %% [markdown]
# Verificamos de nuevo los valores perdidos:

# %%
print('El porcentaje de valores perdidos de las variables: \n')
for var in data.columns:
    print('{} \t\t {:,.2f}%:'.format(var, data[var].isnull().sum()/len(data)*100))

# %% [markdown]
# El 40% de valores perdidos que se observa arriba, corresponde a los datos de test que no contiene esta variale (por ser la variable respuesta que queremos obtener).

# %% [markdown]
# Verificando de nuevo los valores de la variables categóricas:

# %%
print(aux, ':', set(data['Item_Fat_Content']))
print(aux, ':', set(data['Item_Type']))
print(aux, ':', set(data['Outlet_Identifier']))
print(aux, ':', set(data['Outlet_Size']))
print(aux, ':', set(data['Outlet_Location_Type']))
print(aux, ':', set(data['Outlet_Type']))

# %% [markdown]
# Ya se cuenta con un dataset un poco más limpio. Falta verificar las variables numéricas y recodificar las categorias de la variable "Item_Type"; para esta recodificación prodecemos a realizar primero una pruebas de significancia estadísticas. Pero antes, vemos algunos otros comportamientos bivariados:

# %%
for var in ['Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Establishment_Year']:
    plt.figure(figsize=(len(data[var].unique())*2,4))
    sns.violinplot(x=var, y="Item_Outlet_Sales", data=data)
    plt.show()

# %% [markdown]
# Los valores de ventas de las diferentes categorias no tienen distribución normal. Se utilizará el test de Kruskal-Wallis (técnica no paramétrica) para determinar relación significativa entre las distintas variables y los niveles de ventas de la tiendas (TARGET).

# %%
# H0: las medias son significativamente iguales entre los grupos
print('Test de Kruskal-Wallis para Item_Fat_Content vs Item_Outlet_Sales:\n\t', stats.kruskal(list(data.dropna().loc[data['Item_Fat_Content'] == 'Low Fat', 'Item_Outlet_Sales']), 
                                                                                         list(data.dropna().loc[data['Item_Fat_Content'] == 'Regular', 'Item_Outlet_Sales'])))

# H0: las medias son significativamente iguales entre los grupos
print('Test de Kruskal-Wallis para Item_Fat_Content vs Item_Outlet_Sales:\n\t', stats.kruskal(list(data.dropna().loc[data['Outlet_Location_Type'] == 'Tier 1', 'Item_Outlet_Sales']), 
                                                                                         list(data.dropna().loc[data['Outlet_Location_Type'] == 'Tier 2', 'Item_Outlet_Sales']), 
                                                                                         list(data.dropna().loc[data['Outlet_Location_Type'] == 'Tier 3', 'Item_Outlet_Sales'])))

# %% [markdown]
# En los graficos de violin se puede visualizar que el contenido de grasa en los productos no presenta influencia sobre el nivel de ventas y que las otras variables categóricas analizadas si tienen influencia sobre la variable TARGET; esto es corroborado por los test de Kruskal-Wallis realizados.
# 
# Respecto de la variable contenido de grasa de los productos, dicha conclusión arriba hecha no se corresponde con lo que se espera; lo que sugiere revisar más a fondo el registro de estas categorias. Para ello, realicemos una vista general de los datos:

# %%
data[data['Item_Fat_Content'] == 'Low Fat'].head()

# %% [markdown]
# En la 3ra linea se encuentra una inconsistencia; no tiene sentido clasificar como "Low Fat" un producto del hogar. Veamos esto en un gráfico agrupado:

# %%
sns.catplot(y="Item_Type", hue="Item_Fat_Content", kind="count", data=data, height=6, aspect=2)
plt.show()

# %% [markdown]
# Existen productos con categoría "Low Fat" que no son comestibles o que simplemente no tienen ningún contenido de grasa, para ser consistentes se asigna una nueva categoría NA (No aplica) para los tipos de productos Household, Health and Hygiene, Hard Drinks, Soft Drinks, Fruits and Vegetables:

# %% [markdown]
# ## Features Engineering

# %% [markdown]
# #### FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'

# %%
# FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'

data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

sns.catplot(y="Item_Type", hue="Item_Fat_Content", kind="count", data=data, height=6, aspect=2)
plt.show()

# %% [markdown]
# Analicemos los niveles de ventas por contenido de grasa de los productos:

# %%
# H0: las medias son significativamente iguales entre los grupos
stats.kruskal(list(data.dropna().loc[data['Item_Fat_Content']== 'Low Fat', 'Item_Outlet_Sales']), list(data.dropna().loc[data['Item_Fat_Content']== 'Regular', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Item_Fat_Content']== 'NA', 'Item_Outlet_Sales']))  

# %% [markdown]
# No se evidencia diferencias significativas en los niveles de ventas entre las 3 categorias de la característica Item_Fat_Content. Veamos un gráfico de ello:

# %%
sns.violinplot(x="Item_Fat_Content", y='Item_Outlet_Sales', kind="bar", data=data)
plt.show()

# %%
sns.catplot(x="Item_Type", y='Item_Outlet_Sales', hue="Item_Fat_Content", kind="bar", data=data, height=5, aspect=4)
plt.show()

# %% [markdown]
# De forma similar lo vemos en el gráfico por tipo de producto, intentemos reagrupar dichas categoría para buscar una relación significativa con el nivel de ventas.
# 
# Veamos una clasificación por usos:
# - Consultando las categorias de idenificación de los tipos de productos

# %%
print(list(data[data['Item_Type'] == 'Others']['Item_Identifier'].unique()))

# %%
print(list(data[data['Item_Type'] == 'Health and Hygiene']['Item_Identifier'].unique()))

# %%
print(list(data[data['Item_Type'] == 'Household']['Item_Identifier'].unique()))

# %% [markdown]
# En general se observa: FD = ALIMENTOS - NC = HOGAR, SALUD E HIG, OTROS - DR = BEBIDAS, 

# %% [markdown]
#'Others', 'Health and Hygiene', 'Household', 'Baking Goods', 'Breakfast', 'Snack Foods', 'Dairy', 'Fruits and Vegetables', 'Breads', 'Seafood', 'Soft Drinks', 'Starchy Foods', 'Meat', 'Frozen Foods', 'Canned', 'Hard Drinks

#ESPAÑOL:
#'Otros', 'Salud e higiene', 'Hogar', 'Productos para hornear', 'Desayuno', 'Snack Foods', 'Lácteos', 'Frutas y verduras', 'Panes', 'Mariscos', 'Refrescos' , 'Alimentos con almidón', 'Carne', 'Alimentos congelados', 'Enlatados', 'Bebidas Duras

#RECATEGORIZACIÓN SUGERIDA (de acuerdo a la similitud entre los productos):
#1- 'Non perishable':       'Others', 'Health and Hygiene', 'Household'
#2- 'Fruits and Vegetables' 
#3- 'Meats':                'Seafood', 'Meat'
#4- 'Processed Foods':      'Baking Goods', 'Frozen Foods', 'Canned'
#5- 'Starchy Foods':        'Breads', 'Starchy Foods', 'Snack Foods', 'Breakfast'
#6- 'Drinks':               'Soft Drinks', 'Hard Drinks, 'Dairy'

# %% [markdown]
# #### FEATURES ENGINEERING: creando categorías para 'Item_Type'

# %%
# FEATURES ENGINEERING: creando categorías para 'Item_Type'
data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
 'Seafood': 'Meats', 'Meat': 'Meats',
 'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
 'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

# FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

# %% [markdown]
# Visualicemos de nuevo esta recategorización en un gráfico:

# %%
#plt.figure(figsize=(12,4))
#sns.violinplot(x="Item_Type", y='Item_Outlet_Sales', hue="Item_Fat_Content", data=data)
#plt.show()

plt.figure(figsize=(12,4))
sns.violinplot(x='Item_Type', y="Item_Outlet_Sales", data=data)
plt.show()

# %%
data['Item_Type'].unique()

# %%
# H0: las medias son significativamente iguales entre los grupos
stats.kruskal(list(data.dropna().loc[data['Item_Type']== 'Drinks', 'Item_Outlet_Sales']), list(data.dropna().loc[data['Item_Type']== 'Meats', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Item_Type']== 'Fruits and Vegetables', 'Item_Outlet_Sales']), 
             list(data.dropna().loc[data['Item_Type']== 'Non perishable', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Item_Type']== 'Fruits and Vegetables', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Item_Type']== 'Processed Foods', 'Item_Outlet_Sales']),
             list(data.dropna().loc[data['Item_Type']== 'Starchy Foods', 'Item_Outlet_Sales']))  

# %% [markdown]
# No se evidencia diferencias en los niveles de ventas entre las diferentes categorias de tipo de productos (reagrupados).

# %% [markdown]
# ### Análisis Bi-variado:
# Variables Continuas vs Continuas

# %%
numerics_var = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']
sns.pairplot(data.dropna(), x_vars=numerics_var, y_vars=numerics_var, kind='scatter', diag_kind='kde' )
#plt.savefig('hist_scatter')
plt.show()

# %% [markdown]
# No se observa alguna relación significativa entre las variables, lo que señala la necesidad de preprocesar los datos y realizar algunas transformaciones.
# 
# Veamos las correlaciones entre estas variables.

# %%
data[numerics_var].corr()

# %% [markdown]
# - La característica con correlación más alta es Item_MRP (r=0.57), corresponde a una correlación de nivel moderado.
# - El Target guarda una relación casi nula con los pesos de los productos, mientras que con el grado de visibilidad del producto se observa una correlación baja negativa (r=-0.13). Esta última correlación no parece tener sentido, lo que sugiere que estos valores puede que no esten bien registrados.
# - Un aspecto positivo es que la correlación entre las variables independientes es baja, lo que indica que no existe autocorrelación entre estas vraiables.

# %% [markdown]
# #### FEATURES ENGINEERING: Codificando los niveles de precios de los productos

# %%
print(pd.qcut(data['Item_MRP'], 4,).unique())
data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

# %% [markdown]
# ### Codificación de variables ordinales:
# Esta vez no se considera tomar las características: 'Item_Type' y 'Item_Fat_Content'

# %% [markdown]
# Se utiliza una copia de data para separar los valores codificados en un dataframe distinto.

# %%
dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
dataframe.head()

# %% [markdown]
# Se decide realizar una codificación manual y no con algún método automático, para guardar el orden de los valores.  
# 
# Las variables ordinales son: ['Outlet_Size', 'Outlet_Location_Type']

# %%
serie_var = dataframe['Outlet_Size'].unique()
serie_var.sort()
print('Outlet_Size', ':', serie_var)

serie_var = dataframe['Outlet_Location_Type'].unique()
serie_var.sort()
print('Outlet_Location_Type', ':', serie_var)

# %% [markdown]
# #### FEATURES ENGINEERING: Codificación de variables ordinales

# %%
# Codificación de variables ordinales
dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
dataframe.head()

# %% [markdown]
# #### FEATURES ENGINEERING: Codificación de variables nominales

# %%
dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])
dataframe.head()

# %%
print(dataframe.info())

# %% [markdown]
# Revisamos los valores de correlación:

# %%
mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,12))
sns.heatmap(dataframe.corr(), mask=mask, vmax=.3, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .6})
plt.show()

# %% [markdown]
# - El coeficiente de correlación entre las variables independientes es entre bajo y medio, lo que indica que no existe autocorrelación fuerte entre estas variables.

# %% [markdown]
# ### Preparando data de entrenamiento y de test

# %%
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

# %%
df_train.head()

# %%
df_test.head()

# %% [markdown]
# #### ENTRENAMIENTO

# %%
# Importando librerías para el modelo
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression

seed = 28
model = LinearRegression()

# División de dataset de entrenaimento y validación
X = df_train.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
x_train, x_val, y_train, y_val = train_test_split(X, df_train['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

# Entrenamiento del modelo
model.fit(x_train,y_train)

# Predicción del modelo ajustado para el conjunto de validación
pred = model.predict(x_val)

# Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
R2_train = model.score(x_train, y_train)
print('Métricas del Modelo:')
print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

mse_val = metrics.mean_squared_error(y_val, pred)
R2_val = model.score(x_val, y_val)
print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

print('\nCoeficientes del Modelo:')
# Constante del modelo
print('Intersección: {:.2f}'.format(model.intercept_))

# Coeficientes del modelo
coef = pd.DataFrame(x_train.columns, columns=['features'])
coef['Coeficiente Estimados'] = model.coef_
print(coef, '\n')
coef.sort_values(by='Coeficiente Estimados').set_index('features').plot(kind='bar', title='Importancia de las variables', figsize=(12, 6))

plt.show()

# %% [markdown]
# ## Principales variables utilizadas por el modelo:
# - Con relación directa: Outlet_Type_Supermarket Type3, Item_MRP
# - Con relación inversa: Outlet_Type_Grocery Store, Item_Visibility

# %% [markdown]
# ### Aplicación del modelo en el dataset de test

# %%
# Predicción del modelo ajustado
data_test = df_test.copy()
data_test['pred_Sales'] = model.predict(data_test)
data_test.to_csv('data_test')
data_test.head()


