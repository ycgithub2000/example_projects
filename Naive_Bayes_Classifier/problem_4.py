# -*- coding: utf-8 -*-
"""
@author: Yuyuan Cui
"""

import pandas as pd
import numpy as np
# import math as pymath
from scipy.special import comb
from scipy.stats import nbinom

# get column names
headers = pd.read_csv('README', header=None)
x_cols = (headers.loc[2:, 0]).tolist()

# process training data
X = pd.read_csv('X_train.csv', header=None)
X.columns = x_cols
# X.columns
Y = pd.read_csv('label_train.csv', header=None)
Y.columns = [['Y']]
X['Y'] = Y['Y']


a = 1
b = 1
e = 1
f = 1
N_y_1 = len(X[X['Y']==1])
N_y_0 = len(X[X['Y']==0])
N = len(X)
X_sum_y_1 = X[x_cols][X['Y']==1].sum()
X_sum_y_0 = X[x_cols][X['Y']==0].sum()
# X_sum = X[x_cols].sum()


# set up a parameterized function for posterior calculation
class postProb:
    
    def __init__(self, N, N_y_1, N_y_0, 
                X_sum_y_1, X_sum_y_0, 
                x_cols,
                a, b, e, f):
        self.count = 0
        self.N = N
        self.N_y_1 = N_y_1
        self.N_y_0 = N_y_0
        self.X_sum_y_1 = X_sum_y_1
        self.X_sum_y_0 = X_sum_y_0
        self.x_cols = x_cols
        self.a = a
        self.b = b
        self.e = e
        self.f = f
        self.y_star_given_y_1 = (e + N_y_1)/float(N + e + f)
        self.y_star_given_y_0 = (f + N_y_0)/float(N + e + f)

    # log(  p(y* = y | y_vector)  )
    def log_prior_y_star_y(self, y_star):
        return np.log(self.y_star_given_y_1) if y_star == 1 \
                else np.log(self.y_star_given_y_0) 
    
    # Negative Binomial BN(a, b, x*)
    def neg_binomial(self, a, b, x):
        term_1 = (b/float(b+1))**a
        term_2 = (1/float(b+1))**x
        term_3 = comb(a+x-1, x, exact=False)
        # overflow:
        #term_3 = float(pymath.factorial(a+x-1)) / (pymath.factorial(x)*
        #               pymath.factorial(a-1))
        return term_1*term_2*term_3
    
    # log(  p(x* | y* = y, {x_i: y_i=y})  )
    def log_post_x_star_y_star(self, x_star, y_star):
        log_sum = 0.0
        N  = self.N_y_1 if y_star==1 else self.N_y_0
        Nx = self.X_sum_y_1 if y_star==1 else self.X_sum_y_0
        for col in self.x_cols:
            log_sum += np.log(nbinom.pmf(x_star[col], 
                                  Nx[col] + self.a, 
                                  (N+self.b)/float(N+self.b+1)))
        # print(self.count)
        self.count += 1
        return log_sum
        
    # log(  p(y*=1 | x*, X, y_vector)  ) applied to each row of X_test
    def pred_y_star_1(self, df):
        y_star = 1
        # x_star = {df[col] for col in self.x_cols}
        x_star = {col: df[col] for col in self.x_cols}
        # return self.prior_y_star_y(y_star)
        return self.log_post_x_star_y_star(x_star, y_star) + \
               self.log_prior_y_star_y(y_star)
    
    # log(  p(y*=0 | x*, X, y_vector)  ) applied to each row of X_test
    def pred_y_star_0(self, df):
        y_star = 0
        x_star = {col: df[col] for col in self.x_cols}
        #return self.prior_y_star_y(y_star)
        return self.log_post_x_star_y_star(x_star, y_star) + \
               self.log_prior_y_star_y(y_star)


''' train the naive Bayes classifier '''
infer = postProb(N, N_y_1, N_y_0, X_sum_y_1, X_sum_y_0, x_cols, a, b, e, f)

# print(N, N_y_1, N_y_0, X_sum_y_1, X_sum_y_0, x_cols, a, b, e, f)


X_test = pd.read_csv('X_test.csv', header=None)
X_test.columns = x_cols
Y_test = pd.read_csv('label_test.csv', header=None)
Y_test.columns = [['Y']]
X_test['Y'] = Y_test['Y']



''' make predictions for data in the testing set'''
''' numerical issue: 210 row y=1,  '''
X_test['Y_star_1'] = X_test.apply(infer.pred_y_star_1, axis=1)
X_test['Y_star_0'] = X_test.apply(infer.pred_y_star_0, axis=1)
X_test['y_hat_1'] = (X_test['Y_star_1'] > X_test['Y_star_0'])
common_base = np.exp(X_test['Y_star_1']) + np.exp(X_test['Y_star_0'])
X_test['P(Y=1)'] = np.exp(X_test['Y_star_1'])/common_base
X_test['P(Y=0)'] = np.exp(X_test['Y_star_0'])/common_base



''' 2X2 table for prediction quality measurement '''
print(len(X_test[(X_test['Y']==1)&(X_test['Y_star_1']>X_test['Y_star_0'])]))
print(len(X_test[(X_test['Y']==1)&(X_test['Y_star_1']<=X_test['Y_star_0'])]))
print(len(X_test[(X_test['Y']==0)&(X_test['Y_star_1']>=X_test['Y_star_0'])]))
print(len(X_test[(X_test['Y']==0)&(X_test['Y_star_1']<X_test['Y_star_0'])]))


print('\nNumber of tied cases:', 
      len(X_test[X_test['Y_star_1']==X_test['Y_star_0']]))


''' calculate posterior expectation on lambda '''
E_lambda_1 = {}
E_lambda_0 = {}

for c in x_cols:
    E_lambda_1[c] = []
    E_lambda_0[c] = []
    E_lambda_1[c].append( (X_sum_y_1[c] + a)/float(N_y_1 + b) )
    E_lambda_0[c].append( (X_sum_y_0[c] + a)/float(N_y_0 + b) )

E_lambda_1 = pd.DataFrame.from_dict(E_lambda_1)
E_lambda_0 = pd.DataFrame.from_dict(E_lambda_0)

''' export results '''
X_test.to_csv('result.csv')
E_lambda_1.to_csv('lambda1.csv')
E_lambda_0.to_csv('lambda0.csv')





