#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:49:27 2018

@author: yuranpan
"""

import numpy as np
import pandas as pd
from scipy.io import arff
import math
import matplotlib.pyplot as plt
import os

#os.chdir('/Users/yuranpan/Desktop/Fordham/Data_Mining/Hmk4')


def z_score_normalization(dataset_raw):
    dataset_raw_attributes = dataset_raw.iloc[:,0:-1]
    dataset_raw_label = dataset_raw.iloc[:,-1]
    mean_attributes = dataset_raw_attributes.mean(axis = 0)
    sd_attributes = dataset_raw_attributes.std(axis = 0)
    zscore_attributes = (dataset_raw_attributes - mean_attributes)/sd_attributes
    dataset = pd.concat([zscore_attributes, dataset_raw_label], axis = 1 )
    return dataset


def get_initial_centroids(dataset_x,k,random_instances, t):
    init_centroids_matrix = random_instances[0:25*k].reshape(25,k)
    centroid_indexes = init_centroids_matrix[t,:]
    init_centroids_x = dataset_x[centroid_indexes,:]
    return init_centroids_x


def euclidean_distance(datapoint_a_x,centroids):   
    distance = np.sum(((centroids - datapoint_a_x)**2),axis = 1)
    return distance


def assign_kmeans_cluster(dataset_x, centroids_x):
    n = dataset_x.shape[0]
    clusters = np.array([])
    for data in range(n):
        euc_distances = euclidean_distance(centroids_x, dataset_x[data, :])
        min_index = np.argmin(euc_distances)
        clusters = np.append(clusters, min_index)     
    return clusters


def assign_kmeans_centroids(dataset_x,clusters,k):
    new_centroids = np.array([])
    d = dataset_x.shape[1]
    for c in range(k):
        indexes = list(np.where(clusters == c))
        new_centroid = np.mean(dataset_x[indexes,:], axis = 1)
        new_centroids = np.append(new_centroids, new_centroid)
    return (new_centroids).reshape(k,d)



def k_means(dataset_x, k, initial_centroids):
    centroids = initial_centroids
    time_iteration = 0
    while time_iteration <= 50:
        clusters = assign_kmeans_cluster(dataset_x, centroids)     
        new_centroids = assign_kmeans_centroids(dataset_x,clusters, k)
        if np.array_equal(centroids, new_centroids):
            break
        else:
            centroids = new_centroids
            time_iteration = time_iteration + 1
    return clusters


def k_means_sse(dataset_x, final_clusters,k): 
    d = dataset_x.shape[1]
    n = dataset_x.shape[0]
    final_centroids_temp = np.array([])
    sum_sse = 0
    for c in range(k): 
        indexes = list(np.where(final_clusters == c))           
        final_c = np.mean(dataset_x[indexes,:], axis = 1)
        final_centroids_temp = np.append(final_centroids_temp, final_c)
    final_centroids = final_centroids_temp.reshape(k,d)
    for data in range(n):
        sse = np.amin(euclidean_distance(dataset_x[data,:],final_centroids))
        sum_sse = sum_sse + sse
    return sum_sse
      
 

# load the dataset
dataset_raw = pd.DataFrame(arff.loadarff("segment.arff")[0])

# normalize the dataset
dataset = z_score_normalization(dataset_raw)
dataset = dataset.fillna(0)
dataset_x = np.array(dataset.iloc[:,0:-1])

random_instances = np.array([775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214, 53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424, 1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483, 65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641, 661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485, 945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427, 1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240, 524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611, 1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686, 854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737, 1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 
                    1472])


################################################

sd_all_k = []
SSE_all_k = []
for k in range(1,13):
    sse_list = []
    for t in range(25):
        initial_centroids = get_initial_centroids(dataset_x,k,random_instances, t)
        final_clusters = k_means(dataset_x, k, initial_centroids)
        sse = k_means_sse(dataset_x, final_clusters, k)
        sse_list.append(sse)
    SSE = np.mean(sse_list)
    sd = np.std(sse_list)
    print ('when k is',k,'SSE is',SSE)
    print('sd is',sd)
    SSE_all_k.append(SSE)
    sd_all_k.append(sd)
print (SSE_all_k, sd_all_k)    


# plot 
k_list =[i for i in range(1,13)]


plt.plot(k_list, SSE_all_k)
plt.errorbar(x = k_list, y = SSE_all_k, yerr = 2* np.array(sd_all_k), color = 'r', capsize = 4, markersize = 2,fmt = '*')
plt.xlabel('k')
plt.ylabel('SSE')

sse_table = pd.DataFrame({"k":k_list,"μ-2σ":(SSE_all_k-2*np.array(sd_all_k)),"μ":SSE_all_k,
                         "μ+2σ":(SSE_all_k+2*np.array(sd_all_k))})

print(sse_table)








