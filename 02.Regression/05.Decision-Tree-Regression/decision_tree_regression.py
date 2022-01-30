# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
# X contains the Position, Level columns
y = dataset.iloc[:, -1].values
# y contains the salary column

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
## DecisionTreeRegressor(random_state=0)

# Predicting a new result
regressor.predict([[6.5]])
## array([150000.])
"""
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
"""

## Decision Tree Regressor model is clearly not the best model to use on a single feature dataset. It's more adapted to dataset with many features, a high dimensional dataset.

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## The decision tree regressor model is really not the best adapted to two dimensional dataset, with only one feature and one dependent variable.
