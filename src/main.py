from sklearn.datasets import *
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data)
print(type(df))
print(df)
print(df.iloc[1])
print(df.index.tolist())
