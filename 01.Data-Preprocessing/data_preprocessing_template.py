# Data Preprocessing Template

# Importing the libraries
import numpy as np # numpy allows us to work with arrays.
import matplotlib.pyplot as plt #  matplotlib allows us to plot charts and graphs.
import pandas as pd # pandas allows us to import the data, also create matrix of features and the dependent variable.

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # iloc stands for locate indexes, x is our feature(or independent) variable.
y = dataset.iloc[:, :-1].values # y is our dependent variable.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
## X_train contains the features of the independent variable of the training set.
## X_test contains YearsExperience column of Data.csv file.
## y_train contains the features of the dependent variable vector of the training set.
## y_test contains the salary column of Data.csv file.
