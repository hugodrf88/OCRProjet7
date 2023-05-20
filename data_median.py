
# fonction simple qui va calculer la médiane de chaque variable d'un dataframe et qui renvoie
# un dataframe d'une ligne

import pandas as pd
import numpy as np

data=pd.read_csv("./data/data_sample.csv ",index_col=0)

df_med= pd.DataFrame(np.nan, index=[0], columns=data.columns)
numerical_columns = data.select_dtypes(include=np.number).columns
df_med[numerical_columns] = df_med[numerical_columns].fillna(data[numerical_columns].median())
categorical_columns = data.select_dtypes(exclude=np.number).columns
df_med[categorical_columns] = df_med[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

df_med=df_med[data.columns]
df_med.to_csv("./data/data_med.csv")