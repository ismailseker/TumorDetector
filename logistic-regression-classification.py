import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.drop(["Unnamed: 32","id"],axis=1,inplace =True)

# print(data.info())

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)

# normalization = bir featureun diğer featurelere üstünlük sağlamamsı için

x = (x_data - x_data.min()) / (x_data.max() - x_data.min())
x = pd.DataFrame(x, columns=x_data.columns, index=x_data.index)

# train-test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

#  random=state =42 burada tutarlılık sağlıyor, aynı yerden split etmemizi sağlıyor..

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T