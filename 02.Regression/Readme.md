**Q. How to select the best regression model?**

The answer is we've to try all of them and using the [R-squared](https://towardsdatascience.com/wth-are-r-squared-and-adjusted-r-squared-7b816eef90d9) coefficient, we compare them and conclude on which one is the best.

Regression Model | Pros | Cons
| :---:| :--: | :---:
Linear Regression  | Works on any size of dataset, gives informations about relevance of features | The Linear Regression Assumptions
Polynomial Regression  | Works on any size of dataset, works very well on non linear problems | Need to choose the right polynomial degree for a good bias/variance tradeoff
SVR | Easily adaptable, works very well on non linear problems, not biased by outliers | Compulsory to apply feature scaling, not well known, more difficult to understand
Decision Tree Regression | Interpretability, no need for feature scaling, works on both linear/nonlinear problems | Poor results on too small datasets, overfitting can easily occur
Random Forest Regression | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees

