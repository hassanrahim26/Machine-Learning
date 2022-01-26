# Data Preprocessing Tools

## Importing the libraries
import numpy as np # numpy allows us to work with arrays.
import matplotlib.pyplot as plt #  matplotlib allows us to plot charts and graphs.
import pandas as pd # pandas allows us to import the data, also create matrix of features and the dependent variable.

## Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # iloc stands for locate indexes, x is our feature(or independent) variable.
y = dataset.iloc[:, -1].values # y is our dependent variable.

print(X)
"""
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
"""

print(y)
"""
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
"""

## Taking care of missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3]) 
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)
"""
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 63777.77777777778]
 ['France' 35.0 58000.0]
 ['Spain' 38.77777777777778 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
"""

## Encoding Categorical Data

### Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough') 
# passthrough means that we indeed want to keep the columns that won't be One Hot Encoded.
X = np.array(ct.fit_transform(X))
# fit transform method will return the new matrix of features of X with the three columns

print(X)
"""
[[1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [0.0 1.0 0.0 30.0 54000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 1.0 0.0 40.0 63777.77777777778]
 [1.0 0.0 0.0 35.0 58000.0]
 [0.0 0.0 1.0 38.77777777777778 52000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 37.0 67000.0]]
"""

### Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) # Here we don't need a numpy array because this is the depenedent variable vector.

print(y)
"""
[0 1 0 0 1 1 0 1 0 1]
"""

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
"""
[[0.0 0.0 1.0 38.77777777777778 52000.0]
 [0.0 1.0 0.0 40.0 63777.77777777778]
 [1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 35.0 58000.0]]
"""

print(X_test)
"""
[[0.0 1.0 0.0 30.0 54000.0]
 [1.0 0.0 0.0 37.0 67000.0]]
"""

print(y_train)
"""
[0 1 0 0 1 1 0 1]
"""

print(y_test)
"""
[0 1]
"""

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
"""
[[0.0 0.0 1.0 -0.1915918438457856 -1.0781259408412427]
 [0.0 1.0 0.0 -0.014117293757057902 -0.07013167641635401]
 [1.0 0.0 0.0 0.5667085065333239 0.6335624327104546]
 [0.0 0.0 1.0 -0.3045301939022488 -0.30786617274297895]
 [0.0 0.0 1.0 -1.901801144700799 -1.4204636155515822]
 [1.0 0.0 0.0 1.1475343068237056 1.2326533634535488]
 [0.0 1.0 0.0 1.4379472069688966 1.5749910381638883]
 [1.0 0.0 0.0 -0.7401495441200352 -0.5646194287757336]]
"""

print(X_test)
"""
[[0.0 1.0 0.0 -1.0000000000000002 -1.0000000000000002]
 [1.0 0.0 0.0 0.9999999999999999 0.9999999999999997]]
"""
