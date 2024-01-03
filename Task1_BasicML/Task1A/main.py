##############
### Task1A ###
##############

## Import libraries: numpy, pandas and scikit-learn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

## Read in the dataset ##
# read from .csv file into dataframe using pandas
data = pd.read_csv('train.csv', encoding='utf-8')
# split dataset into X and y subsets
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

## Perform 10-fold cross validation for ridge regression ##
# shuffle data before splitting into 10 folds
kf = KFold(n_splits=10, random_state=0, shuffle=True)
# define regularization parameters as given in task statement
lambdas =[0.1, 1, 10, 100, 200]
# intialize list for Root Mean Squared Error (RMSE) averaged over 10 folds
avg_rmse = []
# iterate over all lambdas and compute ridge regression for each value
for lam in lambdas:
    rmse_predict = []
    # perform cross validation
    for train, test in kf.split(X):
        # training and test sets of current fold
        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        # compute ridge regression using training data
        regressor = Ridge(alpha=lam)
        regressor.fit(X_train, y_train)
        # compute prediction using test data
        y_predict = regressor.predict(X_test)
        # compute RMSE of current prediction
        rmse_predict.append(np.sqrt(np.mean(np.square(y_test - y_predict))))
    # compute average RMSE over all folds
    avg_rmse.append(np.mean(np.array(rmse_predict)))

# intialize dataframe with the average RMSE values for each lambda value
df = pd.DataFrame(avg_rmse)
# write dataframe to .csv file
df.to_csv('submission.csv', index=False, header=False)
