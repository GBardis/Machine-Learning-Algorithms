# Regression Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting the Regression to the dataset

# Create a new Regressor

# Predicting  a new result
y_pred = regressor.predict(6.5)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff (Regression Model)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regressor results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff (Regression Model)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
