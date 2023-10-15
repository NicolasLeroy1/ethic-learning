import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_train = pd.read_csv('fakedata_train.csv')
df_test = pd.read_csv('fakedata_test.csv')
n = df_train.shape[0]
p = df_train.shape[1] - 2

predictors = ['X' + str(i) for i in range(1, p+1)]
X = df_train[predictors]
S = df_train['S']
y = df_train['y']
X_test = df_test[predictors]
S_test = df_test['S']
y_test = df_test['y']


# Fit a logistic regression model
logit = lm.LogisticRegression()
logit.fit(X, y)


# make a confusion matrix for y
y_pred = logit.predict(X_test)
confusion_matrix_S1 = pd.crosstab(y_pred[S_test==1], y_test[S_test==1],normalize=True)
confusion_matrix_S0 = pd.crosstab(y_pred[S_test==0], y_test[S_test==0],normalize=True)

# false positive rates
print(['True positive rate of S1',confusion_matrix_S1[1][1]/(confusion_matrix_S1[1][1] + confusion_matrix_S1[1][0])])
print(['False Positive rate of S1',confusion_matrix_S1[1][0]/(confusion_matrix_S1[1][0] + confusion_matrix_S1[1][1])])
print(['True positive rate of S0',confusion_matrix_S0[1][1]/(confusion_matrix_S0[1][1] + confusion_matrix_S0[1][0])])
print(['False Positive rate of S0',confusion_matrix_S0[1][0]/(confusion_matrix_S0[1][0] + confusion_matrix_S0[1][1])])

