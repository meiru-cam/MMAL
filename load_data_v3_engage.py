#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import copy
import time
import pickle as pkl

import multiprocessing as mp
from sklearn import preprocessing

from collections import Counter
import csv
import glob
import os
from sklearn.metrics import f1_score

"""
load_data_v3_engage: contains the functions that utilized in experiments, including data loading, voting label, write output file
difference between this file and load_data_v3_new is that this data loading is for EngageMe dataset

functions:
    get_one_data
    load_mnist
    load_all
    load_train_one
    load_train_one_batchs
    load_mnist_batchs
    cross_count
    voting_multi_instance
    voting_multi_all
    compare_accuracy
    write_csv_game
    write_csv_game2_base
    write_csv3
    write_csv_game2
    summary_its
    write_csv_att
    write_csv_preds

useful link for understanding:
    https://towardsdatascience.com/active-learning-sampling-strategies-f8d8ac7037c8
    https://jacobgil.github.io/deeplearning/activelearning
"""

__author__ = "Beryl Zhang, Oggi Rudovic"
__copyright__ = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"

########################## adjustable parameter ##############################

# number of kids for train, validation and test
num_train_kids = 6
num_dev_kids = 4
num_test_kids = 5

# set a small number for debugging
# num_test_kids = num_dev_kids = num_train_kids = 1

num_sec = 10  # number of frames for each sample
# carefully selected children whose data is more balanced
# new_dev_ids = ['p105','p013','p007', 'p009']
# new_test_ids = ['p008', 'p002', 'p005', 'p015', 'p001']

# load the train, validation and test children ids
# new_train_ids = []
# new_dev_ids = []
# new_test_ids = []
#
# with open('../../../test_child.txt') as f:
#     for line in f.readlines():
#         new_test_ids.append(line.split()[0])
#
# with open('../../../train_child.txt') as f:
#     for line in f.readlines():
#         new_train_ids.append(line.split()[0])
#
# with open('../../../dev_child.txt') as f:
#     for line in f.readlines():
#         new_dev_ids.append(line.split()[0])
#
# train_ids = new_train_ids[:num_train_kids]
# dev_ids = new_dev_ids[:num_dev_kids]
# test_ids = new_test_ids[:num_test_kids]

###############################################################################

# number of cores, multi core to speed up the experiments
num_cores = (mp.cpu_count() - 1) // 2

# modalities
FEATURES = ['deepFeatures',
            'imagenetBirds',
            'imagenetFront',
            'openFace',
            'openPoseBirds',
            'openPoseFront']
# ID of filename assiciated with modalities
CSVI = ['1', '2', '2', '3', '4', '4']


def get_one_data(features_selected, student, rand_section=None, ori=False):
    """
    load data of one section from one kid

    :param features_selected:       list, integer that represents the ids of feature type
    :param student:                 string, the student id
    :param rand_section:            not used
    :param ori:                     not used
    :return:
        raw_feature_test:           list, features
        label_class:                list, labels
        int(rand_section):          int, section selected
    """
    index_features = [0, 257, 327, 354, 378]
    with open(student, 'rb') as f:
        raw = pkl.load(f)
        raw = pd.DataFrame(raw)
        raw.sort_values(5, ascending=True, inplace=True)
        data_train = np.array(raw)

        raw_data_train = []
        label_train = []
        uniques = np.unique(data_train[:, 3])
        df = pd.DataFrame(data_train)
        for unique in uniques:
            df_spec = df.loc[df[3] == unique]
            array_spec = np.array(df_spec)
            window_index = 0
            while window_index < len(array_spec) - 30:
                sum_eng = 0
                if array_spec[window_index + 30][5] - array_spec[window_index][5] == 30:
                    for frame_raw in range(window_index, window_index + 30, 3):
                        raw_data_train.append(array_spec[frame_raw])
                        sum_eng = sum_eng + array_spec[frame_raw][-1]
                    # assign label based on 30 frame
                    label_indicator = sum_eng / 10.0
                    if label_indicator < 0.5:
                        label_train.append(0)
                    # elif label_indicator >= 0.5:
                    #     label_train.append(1)
                    else:
                        label_train.append(1)

                    window_index = window_index + 30
                else:
                    window_index += 1

        raw_feature = []
        for i in features_selected:
            raw_feature.append(
                np.array(raw_data_train)[:, index_features[i-1]:index_features[i]].reshape(len(label_train), 10, -1))

    return raw_feature, np.array(label_train), 0


# def test_data_percent(features_selected, student, cvit, which_section):
#     """
#     load test child data, used in test mode only, split into 7/3 for train/test
#     :param features_selected:       list, feature type number, in same order as taggers
#     :param student:                 string, name of the student data file
#     :param cvit:                    int, test random id
#     :param which_section:           int, the section number
#     :return:
#         pool_x                      array, features of training samples
#         label_pool_x                array, labels for training samples
#         test_x                      array, features of test samples
#         label_text_x                array, labels for test samples
#     """
#     pool_x = []
#     test_x = []

#     raw_data_test, label_class = get_one_data(features_selected, student, rand_section=which_section)

#     indexs = np.linspace(0,len(label_class)-1,len(label_class)).astype(int)
#     # np.random.seed(cvit)
#     np.random.shuffle(indexs)
#     split = len(label_class)//10*7

#     for feature_i in range(len(features_selected)):
#         pool_x.append(raw_data_test[feature_i][indexs[:split]])
#         test_x.append(raw_data_test[feature_i][indexs[split:]])


#    return pool_x, label_class[indexs[:split]], test_x, label_class[indexs[split:]]


def load_mnist(num_classes, type, n_need):
    """
    function used to load mnist data, only used ni file simple.py to test with the active learning performance
    refers to the website https://becominghuman.ai/accelerate-machine-learning-with-active-learning-96cea4b72fdb
    :param num_classes              the number of classes
    :param type:                    sting, type of data, whether 3s or original
    :param n_need:                  int, number of samples to load
    :return:
        dataset_x:                  list, features
        dataset_y:                  list, labels
        child_ID_all, section_ID_all -> meaningless, required for csv writing
    """
    data = np.loadtxt('ActiveLearning/mnist/mnist_{}.csv'.format(type), delimiter=',')
    label = data[:, :1]
    # normalize the feature
    feature = (data[:, 1:].astype(float) + 1) / 255.0

    total = None
    cnt = 0

    for i in range(num_classes):
        ll = np.where(np.array(label) == i)[0]
        if len(ll) != 0:
            i_class_upsampled = np.random.choice(ll, size=n_need)
            if total is not None:
                total = np.concatenate((total, i_class_upsampled)).astype(int)
            else:
                total = np.array(i_class_upsampled).astype(int)
            cnt += 1
    dataset_y = label[total].reshape(-1, )
    dataset_y = dataset_y.astype(int)
    dataset_x = feature[total].reshape(len(label[total]), 1, -1)

    child_ID_all = np.zeros(shape=(n_need * num_classes, 1))
    section_ID_all = np.zeros(shape=(n_need * num_classes, 1))
    print('finish loading')
    return [dataset_x], dataset_y, child_ID_all, section_ID_all


def load_all(feature, type, num_class, ori=False):
    """
    load the mixed training data as balanced as possible, saved in folder 'data_slide5'
    Used in training, at the start of each episode, the data is refreshed for environment
    :param feature:                 list, the feature type numbers
    :param type:                    str, 'train' or 'dev' or 'test'
    :param num_class:               int, number of classes for the classification task
    :param ori:                     bool, use the original features or reduced features
    :return:
        X_train_balanced            list, feature contents of the testing samples during training
        y_trian_balanced            list, labels of samples
    """

    all_data = []
    for i in range(len(feature)):
        all_data.append([])

    all_label = []
    child_ID_all = []
    section_ID_all = []

    # start_time = time.time()
    # pool = mp.Pool(num_cores)
    # all_in_one = pool.starmap(get_one_data, [(feature, child) for child in train_ids])
    # all_in_one = Parallel(n_jobs=num_cores-2)(delayed(get_one_data)(mode, feature, child) for child in train_ids) 

    if type == 'train':
        _ids = train_ids
    elif type == 'dev':
        _ids = dev_ids
    elif type == 'test':
        _ids = test_ids
    else:
        raise ValueError('Wrong ids.')

    for child in _ids:
        label_per_id = []
        data_per_id = []
        for i in range(len(feature)):
            data_per_id.append([])
        child_ID_id = []
        section_ID_id = []

        data = get_one_data(feature, child, 0, ori=ori)
        for i in range(len(feature)):
            data_per_id[i] += data[0][i].tolist()
        # all_label.append(data[1])
        label_per_id += data[1].tolist()
        child_ID_id += [child[29:-4]] * len(data[1])
        section_ID_id += [data[2]] * len(data[1])

        n_need = 200
        total = None
        cnt = 0

        for i in range(num_class):
            ll = np.where(np.array(label_per_id) == i)[0]
            if len(ll) != 0:
                i_class_upsampled = np.random.choice(ll, size=n_need)
                if total is not None:
                    total = np.concatenate((total, i_class_upsampled)).astype(int)
                else:
                    total = np.array(i_class_upsampled).astype(int)
                cnt += 1

        # if cnt != num_class:
        # print('child ', child, ' not balanced')
        # raise ValueError('Some classes are missing.')

        for i in range(len(feature)):
            all_data[i].append(np.array(data_per_id[i])[total])
        all_label.append(np.array(label_per_id)[total])
        child_ID_all.append(np.array(child_ID_id)[total])
        section_ID_all.append(np.array(section_ID_id)[total])

        # label_per_id = np.array(label_per_id)
        # print('child {}, # label 0 {}, # label 1 {}'.format(child, sum(label_per_id==0), sum(label_per_id==1)))

    dataset_x = []
    for i in range(len(feature)):
        dataset_x.append(np.concatenate(all_data[i], axis=0))
    dataset_y = np.concatenate(all_label, axis=0)
    child_ID_all = np.concatenate(child_ID_all, axis=0).reshape(-1, 1)
    section_ID_all = np.concatenate(section_ID_all, axis=0).reshape(-1, 1)

    # X_balanced = []
    # for i in range(len(feature)):
    #     X_balanced.append(np.array(dataset_x[i])[total])
    # y_balanced = np.array(dataset_y)[total]

    # child_ID_all = np.array(child_ID_all)[total].reshape(-1,1)
    # section_ID_all = np.array(section_ID_all)[total].reshape(-1,1)
    # return X_balanced, list(y_balanced), child_ID_all, section_ID_all
    return dataset_x, dataset_y, child_ID_all, section_ID_all


def load_trainone(feature, child=0, num_class=0, ori=False):
    """
    load balanced train data from a randomly chosen section of the selected child

    :param feature:                 list, integer that represents the ids of feature type
    :param child:                   string, the student id
    :param num_class:               int, number of classes for the classification task
    :param ori:                     bool, if true, load original features; if false, load pca reduced features
    :return:
        X_train_one_balanced        list, features
        y_train_one_balanced        list, labels
    """

    raw_data_train, label_class, _ = get_one_data(feature, child, ori=ori)

    dataset_x = []
    for i in range(len(feature)):
        dataset_x.append(raw_data_train[i])
    dataset_y = label_class

    n_need = 200
    total = None
    cnt = 0

    for i in range(num_class):
        ll = np.where(np.array(dataset_y) == i)[0]
        if len(ll) != 0:
            i_class_upsampled = np.random.choice(ll, size=n_need)
            if total is not None:
                total = np.concatenate((total, i_class_upsampled)).astype(int)
            else:
                total = np.array(i_class_upsampled).astype(int)

    X_train_one_balanced = []
    for i in range(len(feature)):
        X_train_one_balanced.append(np.array(dataset_x[i])[total])
    y_train_one_balanced = np.array(dataset_y)[total]

    # section_ID_all = [int(section)]*len(total)
    # child_ID_all = [child]*len(total)

    return X_train_one_balanced, list(y_train_one_balanced)  # , child_ID_all, section_ID_all


def load_trainone_batchs(feature, child=0, num_class=0, ori=False):
    """
    load train data of one child in batches at once, return the data in batches,
    each batch is the data collected from one section

    :param feature:                 list, integer that represents the ids of feature type
    :param child:                   string, the student id
    :param ori:                     bool, if true, load original features; if false, load pca reduced features
    :return:
        x_batchs_all:               list, batches of features
        y_batchs_all:               list, batches of labels
    """
    filenames = []
    with open('../../../allkids/{}_files.txt'.format(child)) as f:
        for line in f.readlines():
            filenames.append(line.split()[0])
    f.close()
    sections = []
    [sections.append(int(filename.split('_')[1])) for filename in filenames]
    sections = list(set(sections))

    all_data = []
    x_batchs_all = []
    for i in range(len(feature)):
        all_data.append([])
        x_batchs_all.append([])
    all_label = []
    y_batchs_all = []

    for section in sections:
        data = get_one_data(feature, child, section, ori=ori)
        for i in range(len(feature)):
            all_data[i].append(data[0][i])
        all_label.append(data[1])

    dataset_x = []
    for i in range(len(feature)):
        dataset_x.append(np.concatenate(all_data[i], axis=0))
    dataset_y = np.concatenate(all_label, axis=0)

    n_need = 50
    total = None
    cnt = 0

    for _ in range(10):
        for i in range(num_class):
            ll = np.where(np.array(dataset_y) == i)[0]
            if len(ll) != 0:
                i_class_upsampled = np.random.choice(ll, size=n_need, replace=False)
                if total is not None:
                    total = np.concatenate((total, i_class_upsampled)).astype(int)
                else:
                    total = np.array(i_class_upsampled).astype(int)

        for i in range(len(feature)):
            x_batchs_all[i].append(np.array(dataset_x[i])[total])
        y_batchs_all.append(np.array(dataset_y)[total])

    return x_batchs_all, y_batchs_all


def load_mnist_batchs(n_feature, type, n_need):
    """
    load mnist in batchs
    :param n_feature:       the number modalities, here is should always be 1
    :param type:            not used
    :param n_need:          number of samples in each batch (batch_size)
    :return:
    """
    data = np.loadtxt('ActiveLearning/mnist/mnist_{}.csv'.format(type), delimiter=',')
    label = data[:, :1].reshape(-1, )
    feature = (data[:, 1:].astype(float) + 1) / 255.  #normalize features
    # feature = data[:, 1:].astype(float)

    n_batch = label.shape[0] // n_need
    permutation = np.random.permutation(label.shape[0])
    feature = feature[permutation, :]
    label = label[permutation]

    x_batchs_all = []
    for i in range(len(n_feature)):
        x_batchs_all.append([])
    y_batchs_all = []
    print('n_batch is ', n_batch)
    for i in range(n_batch):
        y_batchs_all.append(label[i * n_need:i * n_need + n_need].reshape(-1, ))
        for f in range(len(n_feature)):
            x_batchs_all[f].append(feature[i * n_need:i * n_need + n_need].reshape(n_need, 1, -1))
    print('finish data loading')
    return x_batchs_all, y_batchs_all


def cross_count(env, preds_train_cross, preds_test_cross, y_multi_train, y_multi_test):
    """
    Count the correctness of each classifier in case of the samples that the other
    classifier is making correct predictions
    Write csv file and save in records of the experiment
    :param env:                     env, environment
    :param preds_train_cross:       array, predictions from each classifier on train data
    :param preds_test_cross:        array, predictions from each classifier on test data
    :param y_multi_train:           array, prediction after voting (model-fusion case) on train data
    :param y_multi_test:           array, prediction after voting (model-fusion case) on test data
    :return:
        None
    """

    for y_train, y_test, y_true in zip(y_multi_train, y_multi_test, env.train_y_all):
        if y_train == y_true:
            env.cross_counts_correct_train[-1][-2] += 1
        if y_test == y_true:
            env.cross_counts_correct_test[-1][-2] += 1

    for j_pred in range(env.cross_counts_correct_train.shape[0] - 1):
        for y_multi, y_true, y_model in zip(y_multi_train, env.train_y_all, preds_train_cross[j_pred]):
            if y_multi == y_true and y_model == y_true:
                env.cross_counts_correct_train[-1][j_pred] += 1
                env.cross_counts_correct_train[j_pred][-2] += 1

            if y_multi == y_true and y_model != y_true:
                env.cross_counts_incorrect_train[-1][j_pred] += 1

            if y_model == y_true and y_multi != y_true:
                env.cross_counts_incorrect_train[j_pred][-2] += 1

    for i_pred in range(env.cross_counts_correct_train.shape[0] - 1):
        for j_pred in range(env.cross_counts_correct_train.shape[0] - 1):
            for y_model1, y_model2, y_true in zip(preds_train_cross[i_pred], preds_train_cross[j_pred],
                                                  env.train_y_all):
                if y_model1 == y_true and y_model2 == y_true:
                    env.cross_counts_correct_train[i_pred][j_pred] += 1
                if y_model1 == y_true and y_model2 != y_true:
                    env.cross_counts_incorrect_train[i_pred][j_pred] += 1

    for j_pred in range(env.cross_counts_correct_test.shape[0] - 1):
        for y_multi, y_true, y_model in zip(y_multi_test, env.test_y_all, preds_test_cross[j_pred]):
            if y_multi == y_true and y_model == y_true:
                env.cross_counts_correct_test[-1][j_pred] += 1
                env.cross_counts_correct_test[j_pred][-2] += 1
            if y_multi == y_true and y_model != y_true:
                env.cross_counts_incorrect_test[-1][j_pred] += 1

            if y_model == y_true and y_multi != y_true:
                env.cross_counts_incorrect_test[j_pred][-2] += 1

    for i_pred in range(env.cross_counts_correct_test.shape[0] - 1):
        for j_pred in range(env.cross_counts_correct_test.shape[0] - 1):
            for y_model1, y_model2, y_true in zip(preds_test_cross[i_pred], preds_test_cross[j_pred], env.test_y_all):
                if y_model1 == y_true and y_model2 == y_true:
                    env.cross_counts_correct_test[i_pred][j_pred] += 1
                if y_model1 == y_true and y_model2 != y_true:
                    env.cross_counts_incorrect_test[i_pred][j_pred] += 1

    env.cross_counts_correct_train[0][-1] = env.episode
    env.cross_counts_correct_test[0][-1] = env.episode
    env.cross_counts_incorrect_train[0][-1] = env.episode
    env.cross_counts_incorrect_test[0][-1] = env.episode

    f = open(env.csv_correct_train, "a")
    np.savetxt(f, env.cross_counts_correct_train, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_incorrect_train, "a")
    np.savetxt(f, env.cross_counts_incorrect_train, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_correct_test, "a")
    np.savetxt(f, env.cross_counts_correct_test, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_incorrect_test, "a")
    np.savetxt(f, env.cross_counts_incorrect_test, delimiter=",")
    f.write('\n')
    f.close()


def voting_multi_instance(method, pred, confidences, num_class):
    """
    get final prediction of a single instance
    :param method:              str, the voting strategy
    :param pred:                array, predicitions from all classifiers
    :param confidences:         array, confidences of predictions from all classifiers
    :return:
        multi_pred              int, final prediction
    """
    if method == 'maj':
        preds = Counter(pred)
        pred_mode = preds.most_common(1)
        multi_pred = pred_mode[0][0]

    elif method == 'wei':
        pred_array = np.array(pred)
        weights = []
        for i in range(num_class):
            weights.append(sum(pred_array[np.where(pred_array == i)]))
        multi_pred = np.argmax(weights)

    elif method == 'conf':
        multi_pred = pred[np.argmax(np.array(confidences))]

    return multi_pred


def voting_multi_all(method, predd_train, predd_test, conf_train, conf_test, num_class, true_train=None, true_test=None):
    """
    get final prediction of multiple samples
    :param method:              str, the voting strategy
    :param predd_train:         array, predicitions from all classifiers on train data
    :param predd_test:          array, predicitions from all classifiers on test data
    :param conf_train:          array, confidences of predictions from all classifiers on train data
    :param conf_test:           array, confidences of predictions from all classifiers on test data
    :param true_train:          array, ground truth of the training data
    :param true_test:           array, ground truth of the test data
    :return:
        y_multi_train:          array, final predictions of train data
        y_multi_test:           array, final predictions of test data
    """

    y_multi_train = []
    y_multi_test = []

    if method == 'maj':
        ### vote via majority
        for i in range(len(predd_train)):
            preds_train = Counter(predd_train[i])
            y_multi_train.append(preds_train.most_common(1)[0][0])

        for i in range(len(predd_test)):
            preds_test = Counter(predd_test[i])
            y_multi_test.append(preds_test.most_common(1)[0][0])

    elif method == 'conf':
        ### vote via conf
        for i in range(len(predd_train)):
            y_multi_train.append(predd_train[i][np.argmax(np.array(conf_train[i]))])
        for i in range(len(predd_test)):
            y_multi_test.append(predd_test[i][np.argmax(np.array(conf_test[i]))])

    elif method == 'maj_test':
        ### vote via majority
        for i in range(len(predd_train)):
            if true_train[i] in predd_train[i]:
                y_multi_train.append(true_train[i])
            else:
                preds_train = Counter(predd_train[i])
                y_multi_train.append(preds_train.most_common(1)[0][0])

        for i in range(len(predd_test)):
            if true_test[i] in predd_test[i]:
                y_multi_test.append(true_test[i])
            else:
                preds_test = Counter(predd_test[i])
                y_multi_test.append(preds_test.most_common(1)[0][0])

    elif method == 'maj_wtest':

        ### vote via weighted strategy
        weights_train = np.zeros((len(predd_train), 3))
        weights_test = np.zeros((len(predd_train), 3))
        predd_train = np.array(predd_train)
        predd_test = np.array(predd_test)
        for i in range(len(predd_train)):
            for j in range(num_class):
                weights_train[i][j] = sum(predd_train[i][np.where(predd_train[i] == j)])

        for i in range(len(predd_test)):
            for j in range(num_class):
                weights_test[i][j] = sum(predd_test[i][np.where(predd_test[i] == j)])

        y_multi_train = np.argmax(weights_train, axis=1)
        y_multi_test = np.argmax(weights_test, axis=1)

    return y_multi_train, y_multi_test


def compare_accuracy(dev_x_all, dev_y_all, model):
    """
    comparting the accuracy of model with different selection strategy, select the method that work better

    :param dev_x_all:                   list, list of the features
    :param dev_y_all:                   list, list of the predicted labels
    :param model:                       list, list of the models trained for different featrue types
    :return:
        methods[np.argmax(accs)]        str, the method that gives highest accuracy
    """

    methods = ['agent', 'rand', 'uncer', 'diver', 'conser', 'lc']
    accs = []
    vot_method = 'maj'
    num_class = model[0].n_classes
    for method in methods:
        predd_test = []
        for i in range(len(model)):
            predd_test.append(model[i].get_predictions_temp(dev_x_all[i], method))

        predd_test = np.array(predd_test).T.tolist()
        _, y_mul_test = voting_multi_all(vot_method, [], predd_test, [], [], num_class)
        accuracy_mul_test = sum(np.equal(dev_y_all, y_mul_test)) / len(dev_y_all)
        accs.append(float("%.3f" % accuracy_mul_test))
        # print('acc new',float("%.3f" % accuracy_mul_test), 'acc old' ,env.acc_before )
    return methods[np.argmax(accs)]


def write_csv_game(episode, csvfile, train, dev, test, counts, rAll,
                   model):  # need the csv_file defined in gamener and the current episode number
    f = open(csvfile, "a")
    writer = csv.DictWriter(
        f, fieldnames=["episode_number",
                       "accuracy_train",
                       "accuracy_dev",
                       "accuracy_test",
                       "f1_train",
                       "f1_dev",
                       "f1_test",
                       "accuracy_multi_train",
                       "f1_multi_train",
                       "accuracy_multi_dev",
                       "f1_multi_dev",
                       "accuracy_multi_test",
                       "f1_multi_test",
                       "count_0",
                       "count_1",
                       "reward_all",
                       "conf_train",
                       "conf_test"
                       ])
    if episode == 0:
        writer.writeheader()

    method = 'maj'
    num_class = model[0].n_classes

    accuracy_train = []
    accuracy_dev = []
    accuracy_test = []

    conf_train = []
    conf_dev = []
    conf_test = []

    conf_all_train = []
    conf_all_dev = []
    conf_all_test = []

    f1_train = []
    f1_dev = []
    f1_test = []

    predd_train = []
    predd_dev = []
    predd_test = []

    for i in range(len(model)):
        accuracy_train.append(float("%.3f" % model[i].test(train[0][i], train[1])))
        accuracy_dev.append(float("%.3f" % model[i].test(dev[0][i], dev[1])))
        accuracy_test.append(float("%.3f" % model[i].test(test[0][i], test[1])))

        conf_all_train.append(model[i].get_confidence(train[0][i]))
        conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(train[0][i]))))
        conf_all_dev.append(model[i].get_confidence(dev[0][i]))
        conf_dev.append(float("%.3f" % np.mean(model[i].get_confidence(dev[0][i]))))
        conf_all_test.append(model[i].get_confidence(test[0][i]))
        conf_test.append(float("%.3f" % np.mean(model[i].get_confidence(test[0][i]))))
        f1_train.append(float("%.3f" % model[i].get_f1_score(train[1], train[0][i])))
        f1_dev.append(float("%.3f" % model[i].get_f1_score(dev[1], dev[0][i])))
        f1_test.append(float("%.3f" % model[i].get_f1_score(test[1], test[0][i])))

        predd_train.append(model[i].get_predictions(train[0][i]))
        predd_dev.append(model[i].get_predictions(dev[0][i]))
        predd_test.append(model[i].get_predictions(test[0][i]))

    predd_train = np.array(predd_train).T.tolist()
    predd_dev = np.array(predd_dev).T.tolist()
    predd_test = np.array(predd_test).T.tolist()

    conf_all_train = np.array(conf_all_train).T.tolist()
    conf_all_dev = np.array(conf_all_dev).T.tolist()
    conf_all_test = np.array(conf_all_test).T.tolist()

    y_multi_train, y_multi_dev = voting_multi_all(method, predd_train, predd_dev, conf_all_train, conf_all_dev, num_class)
    _, y_multi_test = voting_multi_all(method, [], predd_test, [], [], num_class)

    accuracy_multi_train = sum(np.equal(train[1], y_multi_train)) / len(train[1])
    f1_multi_train = f1_score(train[1], y_multi_train, average='weighted')

    accuracy_multi_dev = sum(np.equal(dev[0], y_multi_dev)) / len(dev[1])
    f1_multi_dev = f1_score(dev[1], y_multi_dev, average='weighted')

    accuracy_multi_test = sum(np.equal(test[1], y_multi_test)) / len(test[1])
    f1_multi_test = f1_score(test[1], y_multi_test, average='weighted')

    # if len(model) > 1:
    # cross_count(env, np.array(predd_train).T, np.array(predd_test).T, y_multi_train, y_multi_test)

    writer.writerow({"episode_number": episode,
                     "accuracy_train": accuracy_train,  # because the performance is now the accuracy on test_in_train
                     "accuracy_dev": accuracy_dev,
                     "accuracy_test": accuracy_test,
                     "f1_train": f1_train,
                     "f1_dev": f1_dev,
                     "f1_test": f1_test,
                     "accuracy_multi_train": float("%.3f" % accuracy_multi_train),
                     "f1_multi_train": float("%.3f" % f1_multi_train),
                     "accuracy_multi_dev": float("%.3f" % accuracy_multi_dev),
                     "f1_multi_dev": float("%.3f" % f1_multi_dev),
                     "accuracy_multi_test": float("%.3f" % accuracy_multi_test),
                     "f1_multi_test": float("%.3f" % f1_multi_test),
                     "count_0": counts[0],
                     "count_1": counts[1],
                     "reward_all": np.sum(rAll) / len(rAll),
                     "conf_train": conf_train,
                     "conf_dev": conf_dev,
                     "conf_test": conf_test
                     })
    f.close()


def write_csv_game2_base(episode, csvfile, train, dev, test, model, method='maj', expnum=None, for_csv=None,
                         flag_csv=None, save_dir=''):  # pass in the accuracy of previous episode
    num_class = model[0].n_classes

    f = open(csvfile, "a")
    writer = csv.DictWriter(
        f, fieldnames=["episode_number",
                       "accuracy_train",
                       "accuracy_dev",
                       "accuracy_test",
                       "f1_train",
                       "f1_dev",
                       "f1_test",
                       "accuracy_multi_train",
                       "f1_multi_train",
                       "accuracy_multi_dev",
                       "f1_multi_dev",
                       "accuracy_multi_test",
                       "f1_multi_test",
                       "conf_train",
                       "conf_dev",
                       "conf_test"
                       ])
    if episode == 0:
        writer.writeheader()

    accuracy_train = []
    accuracy_dev = []
    accuracy_test = []

    conf_train = []
    conf_dev = []
    conf_test = []

    conf_all_train = []
    conf_all_dev = []
    conf_all_test = []

    f1_train = []
    f1_dev = []
    f1_test = []

    predd_train = []
    predd_dev = []
    predd_test = []

    # for i in range(len(model)):
    #     predd_dev.append(model[i].get_predictions_temp(dev[0][i], method=select))
    # predd_dev = np.array(predd_dev).T.tolist()
    # _, y_mul_dev = voting_multi_all(method, [], predd_dev, [], [])
    # accuracy_mul_dev = sum(np.equal(dev[1], y_mul_dev))/len(dev[1])
    # print('acc new',float("%.3f" % accuracy_mul_dev), 'acc old', acc_before )

    # if float("%.3f" % accuracy_mul_dev) > (acc_before-0.02) or episode < 5:
    #     acc_before = float("%.3f" % accuracy_mul_dev)
    #     for i in range(len(model)):
    #         model[i].temp_update(method=select)
    #     update = True
    # else:
    #     update = False
    marg_train = np.zeros((len(train[1]), num_class * len(model)))
    marg_dev = np.zeros((len(dev[1]), num_class * len(model)))
    marg_test = np.zeros((len(test[1]), num_class * len(model)))

    predd_dev = []
    for i in range(len(model)):
        accuracy_train.append(float("%.3f" % model[i].test(train[0][i], train[1])))
        accuracy_dev.append(float("%.3f" % model[i].test(dev[0][i], dev[1])))
        accuracy_test.append(float("%.3f" % model[i].test(test[0][i], test[1])))

        conf_all_train.append(model[i].get_confidence(train[0][i]))
        conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(train[0][i]))))
        conf_all_dev.append(model[i].get_confidence(dev[0][i]))
        conf_dev.append(float("%.3f" % np.mean(model[i].get_confidence(dev[0][i]))))
        conf_all_test.append(model[i].get_confidence(test[0][i]))
        conf_test.append(float("%.3f" % np.mean(model[i].get_confidence(test[0][i]))))

        f1_train.append(float("%.3f" % model[i].get_f1_score(train[1], train[0][i])))
        f1_dev.append(float("%.3f" % model[i].get_f1_score(dev[1], dev[0][i])))
        f1_test.append(float("%.3f" % model[i].get_f1_score(test[1], test[0][i])))

        marg_train[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(train[0][i])
        marg_dev[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(dev[0][i])
        marg_test[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(test[0][i])

        predd_train.append(model[i].get_predictions(train[0][i]))
        predd_dev.append(model[i].get_predictions(dev[0][i]))
        predd_test.append(model[i].get_predictions(test[0][i]))

    predd_train = np.array(predd_train).T.tolist()
    predd_dev = np.array(predd_dev).T.tolist()
    predd_test = np.array(predd_test).T.tolist()

    conf_all_train = np.array(conf_all_train).T.tolist()
    conf_all_dev = np.array(conf_all_dev).T.tolist()
    conf_all_test = np.array(conf_all_test).T.tolist()

    ll = []
    for i in range(len(model)):
        ll.append(1 + i * num_class)
    predd_train_csv = copy.deepcopy(marg_train[:, ll])
    predd_dev_csv = copy.deepcopy(marg_dev[:, ll])
    predd_test_csv = copy.deepcopy(marg_test[:, ll])

    y_multi_train, y_multi_dev = voting_multi_all(method, predd_train, predd_dev, conf_all_train, conf_all_dev,num_class,
                                                  train[1], dev[1])
    _, y_multi_test = voting_multi_all(method, [], predd_test, [], [], num_class, [], test[1])

    accuracy_multi_train = sum(np.equal(train[1], y_multi_train)) / len(train[1])
    f1_multi_train = f1_score(train[1], y_multi_train, average='weighted')

    accuracy_multi_dev = sum(np.equal(dev[1], y_multi_dev)) / len(dev[1])
    f1_multi_dev = f1_score(dev[1], y_multi_dev, average='weighted')

    accuracy_multi_test = sum(np.equal(test[1], y_multi_test)) / len(test[1])
    f1_multi_test = f1_score(test[1], y_multi_test, average='weighted')

    # if len(model) > 1:
    # cross_count(env, np.array(predd_train).T, np.array(predd_test).T, y_multi_train, y_multi_test)

    if flag_csv:
        for (name_list, g_truth, maj_y, pred_l, typed) in zip(for_csv, [train[1], dev[1], test[1]],
                                                              [y_multi_train, y_multi_dev, y_multi_test],
                                                              [predd_train_csv, predd_dev_csv, predd_test_csv],
                                                              ['train', 'dev', 'test']):
            write_csv_preds(name_list[0], name_list[1], np.array(g_truth).reshape(-1, 1),
                            np.array(maj_y).reshape(-1, 1), pred_l, typed, expnum, len(model), csvfile[-5], save_dir)
    writer.writerow({"episode_number": episode,
                     "accuracy_train": accuracy_train,  # because the performance is now the accuracy on test_in_train
                     "accuracy_dev": accuracy_dev,
                     "accuracy_test": accuracy_test,
                     "f1_train": f1_train,
                     "f1_dev": f1_dev,
                     "f1_test": f1_test,
                     "accuracy_multi_train": float("%.3f" % accuracy_multi_train),
                     "f1_multi_train": float("%.3f" % f1_multi_train),
                     "accuracy_multi_dev": float("%.3f" % accuracy_multi_dev),
                     "f1_multi_dev": float("%.3f" % f1_multi_dev),
                     "accuracy_multi_test": float("%.3f" % accuracy_multi_test),
                     "f1_multi_test": float("%.3f" % f1_multi_test),
                     "conf_train": conf_train,
                     "conf_dev": conf_dev,
                     "conf_test": conf_test
                     })
    f.close()

    # return update, acc_before


def write_csv3(episode, csvfile, train, model, select):  # pass in the accuracy of previous episode

    f = open(csvfile, "a")
    writer = csv.DictWriter(
        f, fieldnames=["episode_number",
                       "accuracy",
                       "f1",
                       "accuracy_multi",
                       "f1_multi",
                       "conf"
                       ])
    if episode == 0:
        writer.writeheader()

    method = 'maj'
    num_class = model[0].n_classes

    accuracy_train = []

    conf_train = []

    conf_all_train = []

    f1_train = []

    predd_train = []

    for i in range(len(model)):
        accuracy_train.append(float("%.3f" % model[i].test(train[0][i], train[1])))

        conf_all_train.append(model[i].get_confidence(train[0][i]))
        conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(train[0][i]))))

        f1_train.append(float("%.3f" % model[i].get_f1_score(train[1], train[0][i])))

        predd_train.append(model[i].get_predictions(train[0][i]))

    predd_train = np.array(predd_train).T.tolist()

    conf_all_train = np.array(conf_all_train).T.tolist()

    y_multi_train, _ = voting_multi_all(method, predd_train, [], [], [], num_class)

    accuracy_multi_train = sum(np.equal(train[1], y_multi_train)) / len(train[1])
    f1_multi_train = f1_score(train[1], y_multi_train, average='weighted')

    writer.writerow({"episode_number": episode,
                     "accuracy": accuracy_train,  # because the performance is now the accuracy on test_in_train
                     "f1": f1_train,
                     "accuracy_multi": float("%.3f" % accuracy_multi_train),
                     "f1_multi": float("%.3f" % f1_multi_train),
                     "conf": conf_train
                     })
    f.close()


def write_csv_game2(episode, csvfile, train, dev, test, counts, rAll, model, select, acc_before, acc_att,
                    att_agent=None, method='maj', for_att_csv=None, expnum=None,
                    flag_csv=False, save_dir=''):  # pass in the accuracy of previous episode
    num_class = model[0].n_classes

    if episode < 0:
        att_agent = None

    f = open(csvfile, "a")
    writer = csv.DictWriter(
        f, fieldnames=["episode_number",
                       "accuracy_train",
                       "accuracy_dev",
                       "accuracy_test",
                       "f1_train",
                       "f1_dev",
                       "f1_test",
                       "accuracy_multi_train",
                       "f1_multi_train",
                       "accuracy_multi_dev",
                       "f1_multi_dev",
                       "accuracy_multi_test",
                       "f1_multi_test",
                       "acc_att",
                       "count_0",
                       "count_1",
                       "reward_all",
                       "conf_train",
                       "conf_dev",
                       "conf_test"
                       ])
    if episode == 0:
        writer.writeheader()

    accuracy_train = []
    accuracy_dev = []
    accuracy_test = []

    conf_train = []
    conf_dev = []
    conf_test = []

    conf_all_train = []
    conf_all_dev = []
    conf_all_test = []

    f1_train = []
    f1_dev = []
    f1_test = []

    predd_train = []
    predd_dev = []
    predd_test = []

    # for i in range(len(model)):
    #     predd_dev.append(model[i].get_predictions_temp(dev[0][i], method=select))
    # predd_dev = np.array(predd_dev).T.tolist()
    # _, y_mul_dev = voting_multi_all(method, [], predd_dev, [], [])
    # accuracy_mul_dev = sum(np.equal(dev[1], y_mul_dev))/len(dev[1])
    # print('acc new',float("%.3f" % accuracy_mul_dev), 'acc old', acc_before)

    # if float("%.3f" % accuracy_mul_dev) > (acc_before-0.01) or episode < 5:
    #     acc_before = float("%.3f" % accuracy_mul_dev)
    #     for i in range(len(model)):
    #         model[i].temp_update(method=select)
    #     update = True
    # else:
    #     update = False
    for i in range(len(model)):
        model[i].temp_update(method=select)
    update = True

    predd_dev = []

    if att_agent:
        print('start attention in episode ', episode)
        marg_train = np.zeros((len(train[1]), num_class * len(model)))
        marg_dev = np.zeros((len(dev[1]), num_class * len(model)))
        marg_test = np.zeros((len(test[1]), num_class * len(model)))

    for i in range(len(model)):
        accuracy_train.append(float("%.3f" % model[i].test(train[0][i], train[1])))
        accuracy_dev.append(float("%.3f" % model[i].test(dev[0][i], dev[1])))
        accuracy_test.append(float("%.3f" % model[i].test(test[0][i], test[1])))

        conf_all_train.append(model[i].get_confidence(train[0][i]))
        conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(train[0][i]))))
        conf_all_dev.append(model[i].get_confidence(dev[0][i]))
        conf_dev.append(float("%.3f" % np.mean(model[i].get_confidence(dev[0][i]))))
        conf_all_test.append(model[i].get_confidence(test[0][i]))
        conf_test.append(float("%.3f" % np.mean(model[i].get_confidence(test[0][i]))))

        f1_train.append(float("%.3f" % model[i].get_f1_score(train[1], train[0][i])))
        f1_dev.append(float("%.3f" % model[i].get_f1_score(dev[1], dev[0][i])))
        f1_test.append(float("%.3f" % model[i].get_f1_score(test[1], test[0][i])))

        predd_train.append(model[i].get_predictions(train[0][i]))
        predd_dev.append(model[i].get_predictions(dev[0][i]))
        predd_test.append(model[i].get_predictions(test[0][i]))

        if att_agent:
            marg_train[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(train[0][i])
            marg_dev[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(dev[0][i])
            marg_test[:, i * num_class:i * num_class + num_class] = model[i].get_marginal(test[0][i])

    predd_train = np.array(predd_train).T.tolist()
    predd_dev = np.array(predd_dev).T.tolist()
    predd_test = np.array(predd_test).T.tolist()

    y_att_train = (predd_train == np.reshape(train[1], (-1, 1))).astype(int)
    y_att_dev = (predd_dev == np.reshape(dev[1], (-1, 1))).astype(int)
    y_att_test = (predd_test == np.reshape(test[1], (-1, 1))).astype(int)

    conf_all_train = np.array(conf_all_train).T.tolist()
    conf_all_dev = np.array(conf_all_dev).T.tolist()
    conf_all_test = np.array(conf_all_test).T.tolist()

    # predd_train_csv = copy.deepcopy(predd_train)
    # predd_dev_csv = copy.deepcopy(predd_dev)
    # predd_test_csv = copy.deepcopy(predd_test)
    ll = []
    for i in range(len(model)):
        ll.append(1 + i * num_class)
    predd_train_csv = copy.deepcopy(marg_train[:, ll])
    predd_dev_csv = copy.deepcopy(marg_dev[:, ll])
    predd_test_csv = copy.deepcopy(marg_test[:, ll])

    if att_agent:
        # first apply attention to train val and test
        x_att_train = np.hstack((marg_train, conf_all_train))
        x_att_dev = np.hstack((marg_dev, conf_all_dev))
        x_att_test = np.hstack((marg_test, conf_all_test))

        att_train = att_agent.get_predictions(x_att_train)
        att_dev = att_agent.get_predictions(x_att_dev)
        att_test = att_agent.get_predictions(x_att_test)

        # obtain new:
        for i in range(len(predd_train)):
            if not (att_train[i] == 0).all():
                predd_train[i] = np.array(predd_train[i])[att_train[i] == 1]
        for i in range(len(predd_dev)):
            if not (att_dev[i] == 0).all():
                predd_dev[i] = np.array(predd_dev[i])[att_dev[i] == 1]
        for i in range(len(predd_test)):
            if not (att_test[i] == 0).all():
                predd_test[i] = np.array(predd_test[i])[att_test[i] == 1]

    y_multi_train, y_multi_dev = voting_multi_all(method, predd_train, predd_dev, [], [], num_class)
    _, y_multi_test = voting_multi_all(method, [], predd_test, [], [], num_class)

    accuracy_multi_train = sum(np.equal(train[1], y_multi_train)) / len(train[1])
    f1_multi_train = f1_score(train[1], y_multi_train,  average='weighted')

    accuracy_multi_dev = sum(np.equal(dev[1], y_multi_dev)) / len(dev[1])
    f1_multi_dev = f1_score(dev[1], y_multi_dev,  average='weighted')

    accuracy_multi_test = sum(np.equal(test[1], y_multi_test)) / len(test[1])
    f1_multi_test = f1_score(test[1], y_multi_test,  average='weighted')

    # if len(model) > 1:
    # cross_count(env, np.array(predd_train).T, np.array(predd_test).T, y_multi_train, y_multi_test)

    if flag_csv:
        for (name_list, y_att, pred_att, typed) in zip(for_att_csv, [y_att_train, y_att_dev, y_att_test],
                                                       [att_train, att_dev, att_test], ['train', 'dev', 'test']):
            write_csv_att(name_list[0], name_list[1], y_att, pred_att, typed, expnum, len(model), csvfile[-5], save_dir)
        for (name_list, g_truth, maj_y, pred_l, typed) in zip(for_att_csv, [train[1], dev[1], test[1]],
                                                              [y_multi_train, y_multi_dev, y_multi_test],
                                                              [predd_train_csv, predd_dev_csv, predd_test_csv],
                                                              ['train', 'dev', 'test']):
            write_csv_preds(name_list[0], name_list[1], np.array(g_truth).reshape(-1, 1),
                            np.array(maj_y).reshape(-1, 1), pred_l, typed, expnum, len(model), csvfile[-5], save_dir)

    writer.writerow({"episode_number": episode,
                     "accuracy_train": accuracy_train,  # because the performance is now the accuracy on test_in_train
                     "accuracy_dev": accuracy_dev,
                     "accuracy_test": accuracy_test,
                     "f1_train": f1_train,
                     "f1_dev": f1_dev,
                     "f1_test": f1_test,
                     "accuracy_multi_train": float("%.3f" % accuracy_multi_train),
                     "f1_multi_train": float("%.3f" % f1_multi_train),
                     "accuracy_multi_dev": float("%.3f" % accuracy_multi_dev),
                     "f1_multi_dev": float("%.3f" % f1_multi_dev),
                     "accuracy_multi_test": float("%.3f" % accuracy_multi_test),
                     "f1_multi_test": float("%.3f" % f1_multi_test),
                     "acc_att": float("%.3f" % acc_att),
                     "count_0": counts[0],
                     "count_1": counts[1],
                     "reward_all": float("%.3f" % (np.sum(rAll) / len(rAll))),
                     "conf_train": conf_train,
                     "conf_dev": conf_dev,
                     "conf_test": conf_test
                     })
    f.close()

    return update, acc_before


##############################################################################

def summary_its(expnum, save_dir):
    """
    write the summary file for an experiment
    :param expnum: the list of expeiment numbers
    """

    path = save_dir + 'results/EXP_{}/records/'.format(expnum)
    csvname = path + 'exp{}_summary.csv'.format(expnum)
    files = glob.glob(path + 'exp{}_model*'.format(expnum))

    last = []
    for file in files:
        data = pd.read_csv(file)
        last.append(data.iloc[[-1]])

    df_new = pd.concat(last, axis=0, ignore_index=True)

    single_ = ['accuracy_multi_train', 'f1_multi_train', 'accuracy_multi_dev',
               'f1_multi_dev', 'accuracy_multi_test', 'f1_multi_test']  # , 'count_0', 'count_1']

    single_mean = []
    for i in single_:

        single_mean.append(np.mean(df_new[[i]].values))

    mean = data.iloc[[-1]]

    # for i, it in enumerate(single_):
    mean.loc[:, single_] = single_mean

    df_new = pd.concat([df_new, mean], axis=0, ignore_index=True)
    df_new[single_].reset_index().to_csv(csvname, index=False, header=True, decimal='.', sep=',', float_format='%.3f')


def write_csv_att(child_id, section, att_true, att_pred, typed, expnum, n_fea, ite, save_dir):
    """
    generate results of attention agents
    :param child_id:            array, list in shape of (N,1), the child_id of all samples
    :param section:             array, list in shape of (N,1), the session number of all sampels
    :param att_true:            array, list in shape of (N,M), the true label of samples input to attention agents
    :param att_pred:            array, list in shape of (N,M), the predicted label of samples input to attention agents
    :param typed:                string, 'train' or 'dev' or 'test'
    :param expnum:              int, the experiment number
    :param n_fea:               int, number of features
    :param ite:                int, iter within experiment
    :return:
    """
    csv_file = save_dir + "results/EXP_{0}/records/exp{0}_ite_{2}_att_pred_{1}.csv".format(expnum, typed, ite)
    # if os.path.exists(csv_file):
    #     data = pd.read_csv(csv_file, header=True).as_matrix()
    #     # data[:,1:].astype(int)
    #     new = np.concatenate((child_id, section, att_true, att_pred),axis=1)
    #     data_new = np.concatenate((data, new), axis=0)    # else:
    data_new = np.concatenate((child_id, section, att_true, att_pred), axis=1)
    data_new[:, 1:] = (data_new[:, 1:].astype(float)).astype(int)

    headers = ['child_id', 'section']
    for i in range(n_fea):
        headers.append('L{}'.format(i))
    for i in range(n_fea):
        headers.append('A{}'.format(i))
    df = pd.DataFrame(data_new, columns=headers)
    df.to_csv(csv_file, index=False, header=True)


def write_csv_preds(child_id, section, ground_truth, maj_y, model_pred, typed, expnum, n_fea, ite, save_dir):
    """
    generate results of attention agents
    :param child_id:            array, list in shape of (N,1), the child_id of all samples
    :param section:             array, list in shape of (N,1), the session number of all sampels
    :param ground_truth:        array, list in shape of (N,1), the ground truth label of samples
    :param maj_y:               array, list in shape of (N,1), the final prediction based on majority voting
    :param model_pred:          array, list in shape of (N,M), the predicted label of samples
    :param typed:                string, 'train' or 'dev' or 'test'
    :param expnum:              int, the experiment number
    :param n_fea:               int, number of features
    :param ite:                 int, iter within experiment
    """
    csv_file = save_dir + "results/EXP_{0}/records/exp{0}_ite_{2}_preds_{1}.csv".format(expnum, typed, ite)
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file, header=True).as_matrix()
        # data[:,1:].astype(int)
        new = np.concatenate((child_id, section, ground_truth, maj_y, model_pred), axis=1)
        data_new = np.concatenate((data, new), axis=0)
    else:
        data_new = np.concatenate((child_id, section, ground_truth, maj_y, model_pred), axis=1)
    data_new[:, 4:] = np.round(data_new[:, 4:].astype(float), 3)
    data_new[:, 1:4] = (data_new[:, 1:4].astype(float)).astype(int)

    headers = ['child_id', 'section', 'ground_truth', 'majority']
    for i in range(n_fea):
        headers.append('M{}'.format(i))
    df = pd.DataFrame(data_new, columns=headers)
    df.to_csv(csv_file, index=False, header=True)

if __name__ == 'main':
    summary_its('1', save_dir='D:AL_mnist/')