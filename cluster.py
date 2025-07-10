# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:04:27 2024

@author: Emre AKDEMIR
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#K Means
class cluster_kmeans(object):
    def __init__(self, opt):
        opt.syn_num_closest = int(opt.syn_num * opt.k * opt.percentile_closest / 100)
        self.syn_num_closest = opt.syn_num_closest 
        self.k = opt.k 
        self.percentile_closest = opt.percentile_closest
        self.pca = opt.pca

    def filters(self, syn_feature, syn_label):
        if(self.pca):
            pca = PCA(n_components = 0.95)
            syn_feature_2D = pca.fit_transform(syn_feature)
            kmeans = KMeans(n_clusters=self.k)
            pred = kmeans.fit_predict(syn_feature_2D)
            x_dist = kmeans.transform(syn_feature_2D)
            x_cluster_dist = x_dist[np.arange(len(syn_feature_2D)), kmeans.labels_]
        else:
            kmeans = KMeans(n_clusters=self.k, init='random')
            pred = kmeans.fit_predict(syn_feature)
            x_dist = kmeans.transform(syn_feature)
            x_cluster_dist = x_dist[np.arange(len(syn_feature)), kmeans.labels_]
        
        for i in range(self.k):
            in_cluster = (kmeans.labels_ == i)
            cluster_dist = x_cluster_dist[in_cluster]
            cutoff_distance = np.percentile(cluster_dist, self.percentile_closest)
            above_cutoff = (x_cluster_dist > cutoff_distance)
            x_cluster_dist[in_cluster & above_cutoff] = -1  
        
        filter_syn_feature = torch.rand(self.syn_num_closest, syn_feature.size(dim=1))
        filter_syn_label = torch.rand(self.syn_num_closest) # , syn_label.size(dim=1))
              
        index = 0;
        for i in range(x_cluster_dist.size):
            if(x_cluster_dist[i] != -1):
                filter_syn_feature[index] = syn_feature[i]
                filter_syn_label[index] = syn_label[i]
                index = index + 1  
        return filter_syn_feature, filter_syn_label
    
    

#DBSCAN
class cluster_dbscan(object):
    def __init__(self, opt):
        self.syn_num_closest = opt.syn_num_closest 
        self.k = opt.k 
        self.percentile_closest = opt.percentile_closest   
        self.eps = opt.eps   
        self.min_samples = opt.min_samples   

    def filters(self, syn_feature, syn_label):      
        dbscan = DBSCAN(eps = self.eps, min_samples = self.min_samples)
        dbscan.fit(syn_feature)
        
        indices = dbscan.core_sample_indices_
        filter_syn_feature = syn_feature[indices]
        filter_syn_label = syn_label[indices]
        
        return filter_syn_feature, filter_syn_label