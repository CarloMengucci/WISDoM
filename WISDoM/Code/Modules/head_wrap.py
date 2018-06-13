#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:03:29 2017

@author: carlomengucci
"""


### Subj info file wrapping functions ###

import numpy as np

# %%

## Casistic selection wrapper, takes .csv info file as arguments, and a selection
## parameter for AD; NC; MCI

def wrap_selection (subj_info, sel):

# Header cleaning, cutting suspicious entries
    subj_info_safe=subj_info['safe']==True
    subj_info=subj_info[subj_info_safe]

    if sel == 'AD':
        idx=subj_info['diagnosis']== 'AD'
        idx1=subj_info['Coversion Time'].notnull()
        subj_info=subj_info[(idx | idx1)]
        return (subj_info)

    if sel == 'NC':
        idx=subj_info['diagnosis']== 'NC'
        subj_info=subj_info[idx]
        return (subj_info)

    if sel == 'MCI':
        idx=subj_info['diagnosis']== 'MCI'
        subj_info=subj_info[idx]
        return (subj_info)
# %%

## Extract inferior index of the database sub section to use ##
## (i.e if i want to study cases of the last 200 entries, pass this value
## to case_index and then use to build the tensor)

def sub_data_sel(data, case_index):

    for i in range (len(data.columns)):
        if data.columns[i]>case_index:
            return (i)
# %%

## Designated origin subject index extractor for ABIDE databases.
## A string containing origin identification letters must be passed as argument

def origin_index (origin, data):
    origin_subj_idx = []

    for i in range (len(data['FILE_ID'].values)):
        s=data['FILE_ID'].values[i]
        truth = origin in s

        origin_subj_idx.append(truth)
    return(np.asarray(origin_subj_idx))
