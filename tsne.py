#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tsne: take one thousand frames from each kid and create tsne plot"""

__author__      = "Beryl Zhang, Oggi Rudovic"
__copyright__   = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"

import time
import os
import numpy as np
import pandas as pd
import random

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# Variables for manifold learning.
n_neighbors = 17
n_samples = 1000

############## data loading ##########################333
train_kidsid_all = ['p016', 'p119', 'p205', 'p120', 'p212', 'p124', 'p214', 'p002', 
                    'p117', 'p007', 'p102', 'p012', 'p001', 'p218', 'p013', 'p004', 
                    'p114']

FEATURES = ['deepFeatures', 
            'imagenetBirds',
            'imagenetFront',
            'openFace', 
            'openPoseBirds',
            'openPoseFront']

CSVI = ['1', '2', '2', '3', '4', '4']

############## return n_samples frames of that kid from a random section
def get_one(features_selected, student):
     # get the name of files for the specific test child 
    filenames = []
    with open('../../allkids/{}_files.txt'.format(student)) as f:
        for line in f.readlines():
            l = line.split()[0]
            filenames.append(l)
    f.close()
    
    # if rand_section == None:
    sections = []
    [sections.append(int(filename.split('_')[1])) for filename in filenames]
    sections = list(set(sections))
    rand_section = random.randint(1,8)
    while True:
        if rand_section in sections:
            break
        else:
            rand_section = random.randint(1,8)
            continue

    raw_feature_test = []
    for _ in range(len(features_selected)):
        raw_feature_test.append([])

    for file_i, file_name in enumerate(filenames):
        if int(file_name[5]) == int(rand_section): ### take only one session
            raw_data = []
            mean_label = []
            conf = []
            for feature_i in range(len(features_selected)): 
                data_array = pd.read_csv('../../'+FEATURES[features_selected[feature_i]]+'_pca_500_sd/'+file_name[:-5]+CSVI[features_selected[feature_i]]+'.csv', header = None).as_matrix()[:,3:].astype(float)
                # data_array = pd.read_csv(FEATURES[features_selected[feature_i]]+'/'+file_name[:-5]+CSVI[features_selected[feature_i]]+'.csv', header = None).as_matrix()[:,3:].astype(float)

                raw_data.append(data_array)
                mean_label.append((data_array[:,-3]+data_array[:,-2])/2)
                conf.append(data_array[:,-1])

            count = 0
            ignore_frames = []
            for frame_i in range(len(conf[0])):
                for feature_i in range(len(features_selected)):
                    if conf[feature_i][frame_i] == -1:
                        count += 1
                    if count == len(features_selected):
                        ignore_frames.append(frame_i)
                count = 0
                    
            label_not_missing = []        
            for frame_i in range(len(conf[0])):
                if frame_i in ignore_frames:
                    label_not_missing.append(2000)
                    continue
                else:
                    for fea_i in range(len(features_selected)):
                        if conf[fea_i][frame_i] != -1:
                            label_not_missing.append(mean_label[fea_i][frame_i])
                            break
                    
            label_not_missing = np.array(label_not_missing)
            
            window_index = 0
            #### get raw features
            for fea_i, data_array in enumerate(raw_data):
                for frame_raw in range(len(data_array)):
                    raw_feature_test[fea_i].append(data_array[frame_raw, 1:-3])

    indexs = np.random.choice(list(range(len(raw_feature_test[0]))), size=n_samples)
    for fea_i in range(len(raw_feature_test)):
        raw_feature_test[fea_i] = np.where(raw_feature_test[fea_i] != -1, raw_feature_test[fea_i], 0)
        raw_feature_test[fea_i] = raw_feature_test[fea_i][indexs]
        raw_feature_test[fea_i] = preprocessing.scale(raw_feature_test[fea_i])

    return raw_feature_test


feas = [0,1,2,3,4,5]
data = []
for _ in feas:
    data.append([])

for id_i in train_kidsid_all[:n_neighbors]:
    data_i = get_one(feas,id_i)
    for f_i in range(len(feas)):
        data[f_i].extend(data_i[f_i])

if not os.path.exists('../../plot_tsne'):
    os.makedirs('../../plot_tsne')

for f_i in range(len(feas)):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data[f_i])
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    la = np.zeros((len(data[0]),))
    for i in range(n_neighbors):
        la[i*n_samples:(i+1)*n_samples] += i
    df_subset['y'] = la

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("bright", n_neighbors), # colorblind
        data=df_subset,
        legend="full",
        alpha=0.3)
    plt.savefig('../../plot_tsne/4f_{}.png'.format(f_i))

# # Perform t-distributed stochastic neighbor embedding.
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# trans_data = tsne.fit_transform(sphere_data).T
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))

# ax = fig.add_subplot(2, 5, 10)
# plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

# plt.show()