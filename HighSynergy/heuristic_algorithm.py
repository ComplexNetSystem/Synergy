import pandas as pd
import numpy as np
import networkx as nx
import data_methods as dm
import graph_methods as gm
import community as community_louvain
import matplotlib.pyplot as plt
import powerlaw
import matplotlib.pyplot as plt
import random
from itertools import combinations
from itertools import product
import math
import dit
from sklearn.feature_selection import mutual_info_classif


# Heuristic algorithm: high synergy detection
def ch_net_tri(df_data, top_per, ch_method):
    mi_mat = cal_mi(df_data)
    ch_mat = cond_ent_mat(mi_mat, min_mean_directed=ch_method)
    adj_mat = ch_mat
    threshold = net_mat_threshold(adj_mat, top_per)
    adj_mat[adj_mat < threshold] = 0

    G = nx.from_numpy_matrix(adj_mat)
    all_cliques = list(nx.enumerate_all_cliques(G))
    triadic_cliques = [clq for clq in all_cliques if len(clq) == 3]

    triadic_cliques_tuple = list2tuple_in_list(triadic_cliques)
    tri_clq_df = pd.DataFrame({'comb': triadic_cliques_tuple})
    tri_clq_df.to_csv("triadic_cliques.csv")
    triadic_cliques_df = pd.read_csv("triadic_cliques.csv", index_col=0)

    return triadic_cliques_df


# Finding an appropriate threshold (top_per) in ch_net_tri function. Stavroula will send me the codes...


# Calculate MI Correlation matrix
def cal_mi(df_data):
    num_var = df_data.shape[1]
    mi = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        for rv_t in range(num_var):
            mi[rv_f, rv_t] = mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, rv_f])),
                                                 np.array(df_data.iloc[:, rv_t]),
                                                 discrete_features=True)

    return mi

  
# Calculate condition entropy matrix
def cond_ent_mat(mi, min_mean_directed):
    num_var = len(mi)
    cond_ent_matrix = np.zeros([num_var, num_var])
    for ii in range(num_var):
        for jj in range(num_var):
            # H(X|Y) = H(X) - I(X;Y); H(Y|X) = H(Y) - I(X;Y)
            if ii < jj:
                cond_ent_1 = mi[ii, ii] - mi[ii, jj]
                cond_ent_2 = mi[jj, jj] - mi[ii, jj]
            else:
                continue

            if min_mean_directed == 'min':
                cond_ent_matrix[ii, jj] = np.min([cond_ent_1, cond_ent_2])
                cond_ent_matrix[jj, ii] = np.min([cond_ent_1, cond_ent_2])
            elif min_mean_directed == 'mean':
                cond_ent_matrix[ii, jj] = np.mean([cond_ent_1, cond_ent_2])
                cond_ent_matrix[jj, ii] = np.mean([cond_ent_1, cond_ent_2])
            elif min_mean_directed == 'directed':
                cond_ent_matrix[ii, jj] = cond_ent_2
                cond_ent_matrix[jj, ii] = cond_ent_1

    return cond_ent_matrix
  
  
# Calculate threhold based on top_per
def net_mat_threshold(mat, top_per):
    mat_sorted = sorted(np.reshape(mat, mat.size), reverse=True)
    threshold = mat_sorted[round(mat.size * top_per) - 1]
    return threshold

  
# Data structure transformation
def list2tuple_in_list(list_in_list):
    tuple_in_list = []
    for list_ele in list_in_list:
        tuple_in_list.append(tuple(list_ele))
    return tuple_in_list
  
