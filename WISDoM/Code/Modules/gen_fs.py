#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:29:10 2017

@author: carlomengucci
"""

## Wishart sampling generation wrapper functions ##

import numpy as np
import scipy.stats as st
import pylab as plt
import itertools as it

#%%

## Wraps triu sequences from database into cor matrix to use as scale matrix for
## hypothesis generation

def scale_wrap (x, size):
    c_mat=np.zeros((size,size))
    c_mat[np.triu_indices(size,1)]=x
    c_mat=c_mat.T+c_mat
    np.fill_diagonal(c_mat, 1)
    return (c_mat)
# %%

## Wraps data sequences into matrix and builds complete data tensor ##
## Requires dataframe as data format, and ints as column names for correct indexing ##

def comp_tensor (data, size, start,stop):
    columns=data.columns.values[start:stop]

    tensor=[]

    for i in columns:
        c_mat=np.zeros((size,size))
        c_mat[np.triu_indices(size,1)]=data[i]
        c_mat=c_mat.T+c_mat
        np.fill_diagonal(c_mat, 1)
        tensor.append(c_mat)
    tensor=np.asarray(tensor)
    return (tensor)

# %%

## Sequence generations, takes a wishart.rvs object as x, returns a 1-D seq of values.
## Size arguments defines the size of the sampling to unravel. Defines first axis of the
## tensor containing the simulated cor matrices.

def sequencer(x, size, components):

    c_seq=[]

    if size == 1:
        seq=x[np.triu_indices(components, k=1)]
        c_seq.append(seq)
        c_seq=np.asarray(c_seq)
        c_seq=c_seq.flatten()
        return (c_seq)

    else:
        for i in range (size):
            cor=x[i]
            seq=cor[np.triu_indices(components, k=1)]
            c_seq.append(seq)

        c_seq=np.asarray(c_seq)
        c_seq=c_seq.flatten()
        return(c_seq)


# %%

## Row/Column drop function, estimates the log pdf obtained for the various principal
## submatrices, thus allowing to compute logpdf variations caused by the
## deletion of a node. The variation's distribution for each node gives
## direct measure of that node weight in th entire network.

def RC_drop_var (x, S, dist, df):

    dist=st.wishart(df=df, scale=S)

    s_logpdf=dist.logpdf(x*df)

    ex_logpdf=[]

    for j in range (len(x[0,:])):

        ex_c=np.delete(x, j, axis=0)
        S_c=np.delete(S, j, axis=0)

        ex_c1=np.delete(ex_c, j, axis=1)

        S_c1=np.delete(S_c, j, axis=1)

        dist_c=st.wishart(df=df, scale=S_c1)
        p_ex=dist_c.logpdf(ex_c1*df)

        diff_logpdf= s_logpdf - p_ex

        ex_logpdf.append(diff_logpdf)

    return(np.asarray(ex_logpdf))

# %%
                #### WARNING:OBSOLETE ####
## 2-nd order drop function. The output is no longer a patient-feat map but a matrix
## with the j=i iterations as diagonal elements. The other elements represent the logpdf variation
## for the system computed on the(N-j, N-i) second order principal submatrix.

def RC2_drop_var (x, S, dist, df):

    dist=st.wishart(df=df, scale=S)

    s_logpdf=dist.logpdf(x*df)

    ex_logpdf=np.zeros((len(x[0,:]),len(x[:,0])))

    diag_indices=np.diag_indices(len(x))

    diag=[]

    for j in range (len(x[0,:])):
        for i in range(len(x[:,0])):
            if j == i:        ## Diagonal element list creation, avoids a good number of
                ex_c=np.delete(x, j, axis=0) ## ops and mat copies conflict
                S_c=np.delete(S, j, axis=0)

                ex_c1=np.delete(ex_c, j, axis=1)

                S_c1=np.delete(S_c, j, axis=1)

                dist_c=st.wishart(df=df, scale=S_c1)
                p_ex=dist_c.logpdf(ex_c1*df)

                diff_logpdf= s_logpdf - p_ex

                diag.append(diff_logpdf)
            else:
                ex_c=np.delete(x, j, axis=0)
                S_c=np.delete(S, j, axis=0)

                ex_c1=np.delete(ex_c, j, axis=1)
                S_c1=np.delete(S_c, j, axis=1)
                ############
                ex_c2=np.delete(ex_c1, i-1, axis=0)
                S_c2=np.delete(S_c1, i-1 , axis=0)

                ex_c3=np.delete(ex_c2, i-1, axis=1)
                S_c3=np.delete(S_c2, i-1, axis=1)

                dist_c=st.wishart(df=df, scale=S_c3)
                p_ex=dist_c.logpdf(ex_c3*df)

                diff_logpdf= s_logpdf - p_ex

                ex_logpdf[j,i]=diff_logpdf

    ex_logpdf[diag_indices]=diag

    return(ex_logpdf)

# %%

## Quantile extraction function, use RC_drop_var output maps as arguments,
## performs a column-wise searchsort between the sample map and the data map,
## then a normalization to match system components with quantiles associated to logpdf
## values. In this way a distribution for each component is yielded.

def quantiles_ex (logodd_sample, logodd_data):

    res = plt.empty((len(logodd_data[:,0]), len(logodd_data[0,:])))

    for i in range (len (logodd_data[0,:])):
        v=logodd_sample[:,i]
        v.sort()
        q = np.searchsorted(v, logodd_data[:, i])
        res[:,i]=q/len(logodd_sample[0,:])
    return(res)
# %%

## LogOdd deviation estimation function, estimates deviation over entire
## subject matrix, bypassing the submatrix computation steps

def global_logodd(x, S, dist, df):
        s_logpdf=dist.logpdf(S*df)
        x_logpdf= dist.logpdf(x*df)
        diff_logpdf= s_logpdf - x_logpdf
        return(diff_logpdf)
# %%

### SPD Matrix generator Using Gershgorin Circle Theorem, an array containing ###
### upper triangle values can be passed as *arg, keep in mind that it must
### satisfy the conditions : lenght=((size*size)-size)//2 and |arr_element|<0.25

def rand_SPD (size, diag_val, *args):
    if args:
        tril=args[0]
        c_mat=np.zeros((size,size))
        c_mat[np.triu_indices(size,1)]=tril
        c_mat=c_mat.T+c_mat
        np.fill_diagonal(c_mat, diag_val)

    else:
        rand=plt.randn(100000)
        tril_idx=np.abs(rand)<(0.25)

        tril=rand[tril_idx]
        tril=tril[:((size*size)-size)//2]

        c_mat=np.zeros((size,size))
        c_mat[np.triu_indices(size,1)]=tril
        c_mat=c_mat.T+c_mat
        np.fill_diagonal(c_mat, diag_val)

    return (c_mat)
# %%

### Enhanced RC2 Function using Itertools, reduces operations by a half,
### coupling only on the upper triangle

def RC2_en_drop(x, S, df):

    dist=st.wishart(df=df, scale=S)

    s_logpdf=dist.logpdf(x*df)

    ex_logpdf = np.empty(x.shape)

    size=range(len(x))

    for i, j in it.combinations_with_replacement(size, 2):
        x1 = np.delete(x, [i, j], axis=0)
        x2 = np.delete(x1, [i, j], axis=1)

        S1 = np.delete(S, [i, j], axis=0)
        S2 = np.delete(S1, [i, j], axis=1)

        dist_c=st.wishart(df=df, scale=S2)
        p_ex=dist_c.logpdf(x2*df)

        diff_logpdf= s_logpdf - p_ex

        ex_logpdf[j,i]=diff_logpdf

        ex_logpdf[j, i] = ex_logpdf[i, j]

    return(ex_logpdf)
