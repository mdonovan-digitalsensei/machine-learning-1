import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# replace missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# split the country field into columns we do this so that
# the machine doesn't assume a relationship between country codes

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# change dependent variable to numeric

le = LabelEncoder()
y = le.fit_transform(y)

# splitting the dataset into the training set and test set

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# Feature Scaling
