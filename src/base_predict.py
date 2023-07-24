import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats

# Predicción del modelo ajustado
data_test = df_test.copy()
data_test['pred_Sales'] = model.predict(data_test)
data_test.to_csv('data_test')
data_test.head()