# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:04:27 2024

@author: Emre AKDEMIR
"""


import torch
from sklearn.cluster import DBSCAN


#K DBSCAN
class cluster_dbscan(object):
    def __init__(self, opt):
        self.syn_num_closest = opt.syn_num_closest 
        self.k = opt.k 
        self.percentile_closest = opt.percentile_closest   

    def filters(self, syn_feature, syn_label):      
        dbscan = DBSCAN(eps = 15, min_samples = 10)
        dbscan.fit(syn_feature)
        
        indices = dbscan.core_sample_indices_
        filter_syn_feature = syn_feature[indices]
        filter_syn_label = syn_label[indices]
        
        return filter_syn_feature, filter_syn_label
    
 

