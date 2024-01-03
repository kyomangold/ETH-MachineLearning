##############
### Task1B ###
##############

## Import libraries: numpy, pandas and scikit-learn
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

## Read in the dataset ##
# read from .csv file into dataframe using pandas
data = pd.read_csv('train.csv', encoding='utf-8')
# split dataset into X and y subsets
X = data.iloc[:, 2:]
y = data.iloc[:, 1]

## Define features given in the problem statement ##
# linear
X1 = X
# quadratic
X2 = X**2
# exponential
X3 = np.exp(X)
# cosine
X4 = np.cos(X)
# constant
X5 = np.ones((X.shape[0],1))
# create a set (stack) of feature transformations
phi = np.hstack((X1, X2, X3, X4, X5))

## Prediction calculated as linear function of the features above (linear regression)
lin_reg = RidgeCV(alphas=np.linspace(0.001, 0.01, 5), fit_intercept=False, store_cv_values=True)
lin_reg.fit(phi, y)

# initialize dataframe with coefficients (weights) of the performed linear regression (prediction)
df = pd.DataFrame(lin_reg.coef_)
# write dataframe to .csv file
df.to_csv('submission.csv', index=False, header=False)
