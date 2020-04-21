#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/path/to/Modules')

import gen_fs as gfs

import networkx as nx

### Network Analysis wrapping Utilities ###

## Wrapping function, creates dataframe with edges and nodes
## from ADNI correlation Matrix. The object returned is a DataFrame containing
## an edges list sorted by correlation absolute values.

def sorted_edge_list (seq, size):

    std_mat=gfs.scale_wrap(np.absolute(seq), size)
    np.fill_diagonal(std_mat,0)
    std_d=pd.DataFrame(std_mat)
    idx=std_d.index.values
    col=std_d.columns.values

    val=[]
    pos=[]
    for i in idx:
        for j in col:
            el=std_d[i][j]
            loc=(i,j)
            val.append(el)
            pos.append(loc)

    d={'position':pos, 'val':val}

    to_sort=pd.DataFrame(data=d)

    sorted_AD=to_sort.sort_values(by='val', ascending=False)

    return(sorted_AD)

### Graph builder, uses N edge list's elements to build undirected Graph and assigns feature
## numbers as node labels.

def graph_builder (sorted_AD, N):

    G=nx.Graph()
    edges=sorted_AD['position'][:N].values
    G.add_edges_from(edges)
    l=[n for n in G]
    labels=dict(zip(l,l))

    return(G, labels)

### AVG connected_components size ###

def avg_cc_size (G):
    cc=[c for c in nx.connected_components(G)]

    cc_size=[]

    for i in range(len(cc)):
        cc_len=len(cc[i])
        cc_size.append(cc_len)

    avg_cc_size=np.mean(cc_size)
    return(avg_cc_size)
