#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import scipy.stats as st
import pandas as pd

import matplotlib

import pylab as plt

import os
from os.path import join as pj

import sys
sys.path.insert(0, '/Users/carlomengucci/Desktop/WISDoM/Code/Modules')

import gen_fs as gfs
import head_wrap as hw

import seaborn as sns

import sklearn

## Path Variables ##

data_dir = "/Users/carlomengucci/Desktop/ADNI-Workspace/Data/"
results_dir='/Users/carlomengucci/Desktop/ADNI-Workspace/Results/'


# %%

AD_vs_AD_t_map=pd.read_hdf(pj(results_dir,'final_10_folds_test_map_AD_VS_AD.h5'), 'table')

AD_vs_NC_t_map=pd.read_hdf(pj(results_dir,'final_10_folds_test_map_AD_VS_NC.h5'), 'table')

NC_vs_NC_t_map=pd.read_hdf(pj(results_dir,'final_10_folds_test_map_NC_VS_NC.h5'), 'table')

NC_vs_AD_t_map=pd.read_hdf(pj(results_dir,'final_10_folds_test_map_NC_VS_AD.h5'), 'table')

AD_diff=AD_vs_AD_t_map-NC_vs_AD_t_map

NC_diff= NC_vs_NC_t_map-AD_vs_NC_t_map

AD_diff['label']=1
NC_diff['label']=0

diff_list=[AD_diff, NC_diff]
diff_concat=pd.concat(diff_list, ignore_index=True)



# %% ## Single Feature Logistic Regression

Y=diff_concat['label'].values

X= diff_concat.loc[:, diff_concat.columns != 'label'].values


from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()

a=np.reshape(np.asarray(X[:,1]), (1,len(X[:,1])))

feats=[]

for i in range(549):
    x=np.reshape(np.asarray(X[:,i]), (-1,1))
    LR.fit(x, Y)
    W=LR.coef_[0]
    feats.append(W)

Single_df=pd.DataFrame({'Voxel_ID':np.arange(0,549)+1, 'S_W':feats})

sorted_single=Single_df.sort_values(ascending=False, by='S_W')

sorted_single.head()

## ROC Scoring ###

from sklearn.model_selection import cross_val_score

roc_score=[]

for i in range (549):

    scores = cross_val_score(LR, np.reshape(np.asarray(X[:,i]), (-1,1)),
                                Y,cv=10,scoring='roc_auc')
    score=np.mean(scores)
    roc_score.append(score)

ROC_DF=pd.DataFrame({'Voxel_ID':np.arange(0,549)+1, 'ROC_Score':roc_score})

ROC_sort=ROC_DF.sort_values(by='ROC_Score', ascending=False)

ROC_sort

ROC_sort.to_csv(pj(results_dir,'ROC_Scores.csv'),index=False, header=True, sep='\t' )

f,ax=plt.subplots(figsize=(8,8))
ax.scatter(np.arange(549),ROC_sort['ROC_Score'].values, alpha=0.3)
ax.set_xlabel('Feature Rank', size=20)
ax.set_ylabel('ROC Score', size=20)

ax.set_title('ROC Score Feature Ranking', size=20)
plt.show(f)

# %% Two feature ROC Score
X_1=diff_concat[['46','476']]

scores = cross_val_score(LR, np.reshape(np.asarray(X_1), (-1,2)),
                            Y,cv=10,scoring='roc_auc')
score=np.mean(scores)
score

# %% Two feature LDA Boundaries

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
clf = LDA()
clf.fit(X_1, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-300, 350)
yy = a * xx - (clf.intercept_[0]) / w[1]

# %%
#### LMplot on first 2 features ####

diff_concat.columns=map(str,(diff_concat.columns))

import matplotlib.patches as mpatches

AD_patch = mpatches.Patch(color='orange', label='AD Subjects')
NC_patch = mpatches.Patch(color='blue', label='NC Subjects')
LDA_patch= mpatches.Patch(color='black', label='LDA Decision Boundary')

f=sns.lmplot('46', '476',
data=diff_concat, hue='label', fit_reg=False, legend=False)
plt.gca().set_xlabel('First Ranked Feature', size=15)
plt.gca().set_ylabel('Second Ranked Feature', size=15)
plt.gca().legend(handles=[AD_patch, NC_patch, LDA_patch])
plt.gca().set_title('ROC AUC score=0.84', size=15)
plt.gca().plot(xx, yy, 'k-')


plt.show(f)

# %% ### Clustering ###

# %%
metric='cityblock'
cmap=sns.diverging_palette(145, 10, sep=5, n=5, l=50, s=90)

f1=sns.clustermap((diff_concat.iloc[:,:-1])**1/5, row_colors=diff_concat['label'],
                  cmap=cmap, metric=metric)
f1.ax_heatmap.set_xlabel('Features', fontsize=15)
f1.ax_heatmap.set_ylabel('Subjects', fontsize=15)

plt.show(f1)
