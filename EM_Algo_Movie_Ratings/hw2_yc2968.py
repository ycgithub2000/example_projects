# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import time
from scipy.stats import norm

TRAINING_FILE = 'ratings.csv'
TESTING_FILE = 'ratings_test.csv'

def stepE(sigma, U, V, R, E):
    u_v = U.T.dot(V)
    norm_cdf = norm.cdf(-u_v/sigma, loc=0, scale=1)
    norm_pdf = norm.pdf(-u_v/sigma, loc=0, scale=1)
    r_pos = u_v + np.nan_to_num(sigma*norm_pdf/(1.0-norm_cdf))
    r_neg = u_v - np.nan_to_num(sigma*norm_pdf/norm_cdf)
    E = (R==1)*r_pos + (R==-1)*r_neg  # how to modify E from inside?
    return E

def stepM(df, c, sigma, U, V, E):
    next_U = np.zeros(U.shape)
    next_V = np.zeros(V.shape)
    d = next_U.shape[0]
    N = U.shape[1]
    M = V.shape[1]
    sigma_2 = sigma**2
    
    for i in range(next_U.shape[1]):
        # print('  M-step, i=', i, ' out of', next_U.shape[1])
        j_slice = df['j'][df['i']==i].values
        m_i = len(j_slice)
        if m_i==0:
            continue
        # important to reshape otherwise will throw error when len(j_sclice)==1
        # if j_sclice is not ".values", will throw non-int index warning
        # sum of outer product (for each column of V)
        v_slice = V[:, j_slice].reshape(d, m_i)
        v_v_t = np.einsum('ij,ik->ijk', v_slice.T, v_slice.T)
        sum_v_v_t = v_v_t.sum(axis=0)
        
        # important: not "m_i/c" !
        # u_coef = ((m_i/c)*np.identity(d)) + (sum_v_v_t/sigma_2)
        # wrong: u_coef = ((N/c)*np.identity(d)) + (sum_v_v_t/sigma_2)
        u_coef = ((1.0/c)*np.identity(d)) + (sum_v_v_t/sigma_2)
        try:
            u_coef_inv = np.linalg.inv(u_coef)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('i=', i)
            else:
                raise
        
        # important to reshape otherwise will throw error when len(j_slice)==1
        v_e = (V[:, j_slice] * E[i, j_slice]).reshape(d, m_i)
        v_e_sum = v_e.sum(axis=1)
        
        next_U[:, i] = np.dot(u_coef_inv, v_e_sum/sigma_2)
    
    for j in range(next_V.shape[1]):
        # print('  M-step, j=', j, ' out of', next_V.shape[1])
        i_slice = df['i'][df['j']==j].values
        m_j = len(i_slice)
        if m_j==0:
            continue
        # important to reshape otherwise will throw error when len(i_sclice)==1
        # if i_sclice is not ".values", will throw non-int index warning
        # sum of outer product (for each column of U)
        u_slice = U[:, i_slice].reshape(d, m_j)
        u_u_t = np.einsum('ij,ik->ijk', u_slice.T, u_slice.T)
        sum_u_u_t = u_u_t.sum(axis=0)
        
        # important: not "m_j/c" !
        # v_coef = ((m_j/c)*np.identity(d)) + (sum_u_u_t/sigma_2)
        # wrong: v_coef = ((M/c)*np.identity(d)) + (sum_u_u_t/sigma_2)
        v_coef = ((1.0/c)*np.identity(d)) + (sum_u_u_t/sigma_2)
        try:
            v_coef_inv = np.linalg.inv(v_coef)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('j=', j)
            else:
                raise
        # important to reshape otherwise will throw error when len(i_slice)==1
        u_e = (U[:, i_slice] * E[i_slice, j]).reshape(d, m_j)
        u_e_sum = u_e.sum(axis=1)
        
        next_V[:, j] = np.dot(v_coef_inv, u_e_sum/sigma_2)
    
    return (next_U, next_V)


def calcLogLike(c, sigma, U, V, R, M, N, d, U_adj_size, V_adj_size):
    # ignores constant denominator
    # column-wise dot product
    u_t_u = ((U*U).sum(axis=0))
    # wrong:
    # log_p_u = (-1.0/(2*c)) * (np.dot(U_adj_size, u_t_u))
    log_p_u = (-1.0/(2*c))*(u_t_u.sum())  # log P(U)
    
    v_t_v = ((V*V).sum(axis=0))
    # wrong:
    # log_p_v = (-1.0/(2*c))*(np.dot(V_adj_size, v_t_v))
    log_p_v = (-1.0/(2*c))*(v_t_v.sum())  # log P(V)
    
    u_v = U.T.dot(V)
    norm_cdf = norm.cdf(u_v/sigma, loc=0, scale=1)
    r_pos_u_v = norm_cdf  # P(r_i_j = 1 | u_i, v_j)
    r_neg_u_v = 1.0-norm_cdf  # P(r_i_j = -1 | u_i, v_j)
    
    # log P(R|U,V)
    '''
    if np.min(np.abs(r_pos_u_v))<0.0000000000000000001:
       print('r_pos_u_v small value: ', np.nonzero(np.abs(r_pos_u_v)<0.0000000000000000001))
    
    if np.min(np.abs(r_neg_u_v))<0.0000000000000000001:
       print('r_neg_u_v small value: ', np.nonzero(np.abs(r_neg_u_v)<0.0000000000000000001))
    '''
    log_p_r = np.sum((R==1)*np.nan_to_num(np.log(r_pos_u_v)) 
                     +(R==-1)*np.nan_to_num(np.log(r_neg_u_v)))
    
    log_const = (-(M+N)*d/2.0)*np.log(2*np.pi*c)
    
    return log_p_u + log_p_v + log_p_r + log_const



def makePrediction(df, sigma, U, V):
    # The scale keyword specifies the standard deviation
    df['uv'] = df.apply(lambda x: U[:, x['i']].dot(V[:, x['j']]), axis=1)
    df['p_phi_neg'] = norm.cdf(0, loc = df['uv'], scale=sigma)
    df['p_phi_pos'] = 1-df['p_phi_neg']
    df['r_hat'] = -1*(df['p_phi_neg'] >df['p_phi_pos'])\
                  +1*(df['p_phi_neg']<=df['p_phi_pos'])
    
    res = [len( df[(df['r'] == +1) & (df['r_hat'] == +1)] ),
           len( df[(df['r'] == -1) & (df['r_hat'] == +1)] ),
           len( df[(df['r'] == +1) & (df['r_hat'] == -1)] ),
           len( df[(df['r'] == -1) & (df['r_hat'] == -1)] )]      
    print('r = +1, r_hat = +1 cases:', res[0])
    print('r = -1, r_hat = +1 cases:', res[1])
    print('r = +1, r_hat = -1 cases:', res[2])
    print('r = -1, r_hat = -1 cases:', res[3])
    return res    


df = pd.read_csv(TRAINING_FILE, header=None)
df.columns = ['i', 'j', 'r']
# df.head()

omega_i = set(df['i'])
omega_j = set(df['j'])

MAX_ITER = 100
d = 5
sigma = 1.0
c = 1.0
N = max(df['i']) + 1  # upper bound of i and j's range
M = max(df['j']) + 1
init_mean = np.zeros(d)
init_cov = np.identity(d)*0.1

# U is d-by-N matrix, V is d-by-M matrix
U = np.random.multivariate_normal(init_mean, init_cov, N).T
V = np.random.multivariate_normal(init_mean, init_cov, M).T
U[:, list(set(range(U.shape[1])) - omega_i)] = 0.0
V[:, list(set(range(V.shape[1])) - omega_j)] = 0.0
# R = np.empty((N, M)) * np.nan
R = np.zeros((N, M))
for idx, row in df.iterrows():
    R[row['i'], row['j']] = row['r']
E = np.zeros((N, M))

U_adj_size = np.array([len(df[df['i']==i]) for i in range(N)])
V_adj_size = np.array([len(df[df['j']==j]) for j in range(M)])

log_likelihood = []


res = {'Iter': [], 'Likelihood': []}
t_total = time.time()
for i in range(MAX_ITER):
    t0 = time.time()
    E = stepE(sigma, U, V, R, E)
    print('Iteration', i+1, ': E-step time:', time.time()-t0)
    
    t0 = time.time()
    U, V = stepM(df, c, sigma, U, V, E)
    print('Iteration', i+1, ': M-step time:', time.time()-t0)

    ln_R_U_V = calcLogLike(c, sigma, U, V, R, 
                           M, N, d, U_adj_size, V_adj_size)
    log_likelihood.append(ln_R_U_V)
    
    res['Iter'].append(i+1)
    res['Likelihood'].append(ln_R_U_V)
    print('Iteration', i+1, ': ln P(R, U, V):', ln_R_U_V, '\n')
   
print('Total num of iterations: ', MAX_ITER, '; Total time: ', time.time()-t_total)
res = pd.DataFrame.from_dict(res)

test = pd.read_csv(TESTING_FILE, header=None)
test.columns = ['i', 'j', 'r']

makePrediction(test, sigma, U, V)
makePrediction(df, sigma, U, V)

res.to_csv('03_Iterations.csv')
test.to_csv('03_test_result.csv')
df.to_csv('03_train_result.csv')
