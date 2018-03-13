import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import scipy.interpolate

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training stest

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal modal Backward Elimination
def create_prediction_list(X):
    prediction_list = []
    for i in range(0, len(X[1])):
        prediction_list.append(i)
    return prediction_list


SL = 0.05

X = np.append(arr=np.ones((50, 1)), values=X, axis=1)
prediction_list = create_prediction_list(X)
X_opt = X[:, prediction_list]
prediction_list = create_prediction_list(X_opt)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()



# def remove_predictor(prediction_list, index_to_be_removed=None):
#     if index_to_be_removed is not None:
#         prediction_list.pop(index_to_be_removed)
#         return prediction_list
#     else:
#         return prediction_list
#
#
# count = 0
# while count <= len(regressor_OLS.pvalues):
#     prediction_list_length = len(regressor_OLS.pvalues)
#     for i in range(0, prediction_list_length):
#         if max(regressor_OLS.pvalues).astype(float) > SL:
#             X_opt = X[:, remove_predictor(prediction_list, i)]
#             regressor_OLS = sm.OLS(y, X_opt).fit()
#             print(regressor_OLS.summary())
#             count = 0
#             break
#         else:
#             count = count + 1
#             continue




def backwardElimination(x, sl):
    global regressor_OLS
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    #print(regressor_OLS.summary())
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
