import pandas as pd
import numpy as np
n=1005
p=3

def logit(x):
    return 1/(1+np.exp(-x))

np.random.seed(2)
beta = np.random.randn(p)

X = np.random.randn(n, p)
Xt = np.random.randn(n, p)

S = np.random.binomial(1, 0.2, n)
St = np.random.binomial(1, 0.5, n)

y = np.random.binomial(1, logit(np.random.randn(n) + 0.6 * S + X@beta ), n)
yt = np.random.binomial(1, logit(np.random.randn(n) + 0.6 * St + Xt@beta ), n)

colnames = ['X' + str(i) for i in range(1, p+1)]
df = pd.DataFrame(X, columns=colnames)
dft = pd.DataFrame(Xt, columns=colnames)

df['S'] = S
dft['S'] = St

df['y'] = y
dft['y'] = yt



df.to_csv('fakedata_train.csv', index=False)
dft.to_csv('fakedata_test.csv', index=False)


