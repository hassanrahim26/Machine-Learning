# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =  0)
## X_train contains the features of the independent variable of the training set.
## X_test contains YearsExperience column of Salary_Data.csv file.
## y_train contains the features of the dependent variable vector of the training set.
## y_test contains the salary column of Salary_Data.csv file.

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
## In Regression we've to predict a continuous real value.
## In Classification we've to predict a category.
regressor.fit(X_train, y_train)
## The fit method will train the regression model(simple linear regression model) on the training set.


# Predicting the Test set results
y_pred = regressor.predict(X_test)
## y_pred contains the predicted salaries.

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
## x-axis is the number of years of experience.
## y-axis is the salary.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
## Calling the predict method on X_train, meaning the number of years of experience of the employees in the training set, will give us exactlty the predicted salaries on the training set.
## Regression Line:- The regression line is the line of the predictions coming as close as possible to the real result(salary). The regression line which we get is actually resulting from a unique equation.
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 
## show function will display the graph in the output.

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
## The regression line which we get is actually resulting from a unique equation. Therefore, the predicted salaries of the test set will be on the same regression line as the predicted salaries of the training set. That's why we'll not replace X_train by X_test here.
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))
## [138531.00067138]
"""
Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.
Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:
12->scalar
[12]->1D array
[[12]]->2D array
"""

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
## [9312.57512673]
## 26780.09915062818
"""
Therefore, the equation of our simple linear regression model is:
Salary = 9345.94 × YearsExperience + 26816.19
Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.
"""
