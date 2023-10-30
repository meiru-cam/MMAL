#!/usr/bin/env python
# -*- coding: utf-8 -*-

from diversity_v1 import diversitySampling
from sklearn import preprocessing
import random
import numpy as np
from scipy.stats import mode

"""
apply_filter_v1: the active sampling methods used to select informative samples

functions:
    bald_sampling
    VarRatio
    random_sampling
    uncertainty_sampling
    conservative_sampling
    diversity_sampling
    least_confident

useful link for understanding: 
    https://towardsdatascience.com/active-learning-sampling-strategies-f8d8ac7037c8
    https://jacobgil.github.io/deeplearning/activelearning
"""

__author__ = "Beryl Zhang, Oggi Rudovic"
__copyright__ = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"


################################ Bayesian Dropout #############################
def bald_sampling(model_selected, budget, train_data):
    """
    bayesian dropout as unsertainty estimation -> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]
    
    :return: x_pool_index: the index of selected samples from the input data
    """
    
    dropout_iterations = 100
    
    U_all = np.zeros(shape=(len(train_data[1])))
    for model in range(len(model_selected)):
        score_All = np.zeros(shape=(len(train_data[1]), 3))
        All_Entropy_Dropout = np.zeros(shape=len(train_data[1]))
        for d in range(dropout_iterations):
            # shape of N*3
            dropout_score = model_selected[model].get_marginal(train_data[0][model])
            # computing G_X
            score_All = score_All + dropout_score

            # computing F_X
            # shape of N*3
            dropout_score_log = np.log2(dropout_score)
            # shape of N*3, elementwise multiplication
            Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
            # entropy per sample, sum over different classes
            Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
            # shape of N
            All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout

        Avg_Pi = np.divide(score_All, dropout_iterations)
        Log_Avg_Pi = np.log2(Avg_Pi)
        Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

        G_X = Entropy_Average_Pi

        Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)

        F_X = Average_Entropy

        U_X = G_X - F_X

        U_all = U_all + U_X

    a_1d = U_all.flatten()
    x_pool_index = a_1d.argsort()[-budget:]
    return x_pool_index
#############################################################################


############################ dropout VarRatio ###############################
def VarRatio(model_selected, budget, train_data):
    """
    dropout variational ratio as unsertainty estimation -> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]
    
    :return: x_pool_index: the index of selected samples from the input data
    """

    dropout_iterations = 100
    All_Dropout_Classes = np.zeros(shape=(len(train_data[1]), 1))
    Variation_all = np.zeros(shape=(len(train_data[1])))
    for i in range(len(model_selected)):
        for d in range(dropout_iterations):
            dropout_classes = model_selected[i].get_predictions(train_data[0][i])
            dropout_classes = np.array([dropout_classes]).T
            All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

        Variation = np.zeros(shape=(len(train_data[1])))

        for t in range(len(train_data[1])):
            L = np.array([0])
            for d_iter in range(dropout_iterations):
                L = np.append(L, All_Dropout_Classes[t, d_iter+1])                      
            Predicted_Class, Mode = mode(L[1:])
            v = np.array([1 - Mode/float(dropout_iterations)])
            Variation[t] = v
        Variation_all = Variation_all + Variation

    a_1d = Variation_all.flatten()
    x_pool_index = a_1d.argsort()[-budget:]
    return x_pool_index

##############################################################################


############################ RANDOM SAMPLING ##################################
def random_sampling(model_selected, budget, train_data, cvit=None):
    """
    select the samples randomly -> sampling strategy
    
    :param model_selected:      list of models selected (multimodal approach, each model contribute to one feature)
    :param budget:              the number of samples to select
    :param train_data:          input data, format in list, [[feature1], [feature2], ..., [featureN]]
    :param cvit:                the random seed applied

    :return: query_indexs: the index of selected samples from the input data
    """
    order = np.linspace(0, len(train_data[0][0])-1, len(train_data[0][0]))
    order = order.astype(int)
    
    # the setting used before to restrict the number of samples seen by the random sampling method
    # to keep the experiment same as the RL-approach before
    # because the RL agent will stop once there are enough samples

    # sample_N = min(budget*4,len(train_data[1]))
    # order = np.linspace(0, sample_N-1, sample_N)
    # order = order.astype(int)
     
    budget = min(budget, len(order))
    if cvit:
        random.seed(cvit)
    random.shuffle(order)
    queried_indexs = random.sample(list(order), budget)
    return queried_indexs
###############################################################################


######################### UNCERTAINTY SAMPLING ################################
def uncertainty_sampling(model_selected, budget, train_data):
    """
    1-confidence as the uncertainty, select the samples with highest uncertainty-> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]

    :return: query_indexs: the index of selected samples from the input data
    """

    # compute uncertaity, which is 1-confidence:
    sample_N = len(train_data[1])
    
    budget = min(budget, sample_N)
    uncertainty = np.zeros((sample_N,))
    ones = np.ones((sample_N,))
    
    for i in range(len(model_selected)):
        uncertainty = uncertainty + (ones - model_selected[i].get_confidence(list(train_data[0][i])))
    queried_indexs = sorted(range(len(uncertainty)), key=lambda i: uncertainty[i])[-budget:]
    return queried_indexs
        
###############################################################################


######################### COSERVATIVE SAMPLING ################################       
def conservative_sampling(model_selected, budget, train_data):
    """
    select the samples based on conservative sampling (least margin) -> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]

    :return: query_indexs: the index of selected samples from the input data
    """

    sample_N = len(train_data[1])
    
    budget = min(budget, sample_N)
    confidence = []
    conf_diff = np.zeros((sample_N,))
    
    for i in range(len(model_selected)):
        confidence.append(model_selected[i].get_confidence(list(train_data[0][i])))
    # the max indecies
    ind_max = np.argmax(confidence, axis=0)
    # the min indecies
    ind_min = np.argmin(confidence, axis=0)
    for i in range(sample_N):
        conf_diff[i] = confidence[ind_max[i]][i]-confidence[ind_min[i]][i]
    queried_indexs = sorted(range(len(conf_diff)), key=lambda i: conf_diff[i])[:budget]
    
    return queried_indexs
###############################################################################

        
############################## DIVERSITY SAMPLING #############################
def diversity_sampling(model_selected, budget, train_data, reference=[]):
    """
    select the samples based on diversity between samples, if reference not provided
    if reference not provided use the mean of input samples as reference -> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]
    :param reference: the reference point to compute the diversity

    :return: query_indexs: the index of selected samples from the input data
    """
    sample_N = len(train_data[0][0])
    
    budget = min(budget, sample_N)
    train_inside = np.concatenate(train_data[0], axis=2)
   
    to_change = np.mean(train_inside, axis=1)
    x_train = preprocessing.normalize(to_change, axis=0)
    
    s = diversitySampling(x_train, pool=reference, budget=budget)
    s.updateCplus()
    queried_indexs = s.newind
    return queried_indexs
###############################################################################


############################# LEAST CONFIDENT #################################
def least_confident(model_selected, budget, train_data):
    """
    select the samples with smallest confidence -> sampling strategy
    
    :param model_selected: list of models selected (multimodal approach, each model contribute to one feature)
    :param budget: the number of samples to select
    :param train_data: input data, format in list, [[feature1], [feature2], ..., [featureN]]

    :return: query_indexs: the index of selected samples from the input data
    """

    sample_N = len(train_data[1])
    
    budget = min(budget, sample_N)
    marginal = []
    ones = np.ones((sample_N,))
    for i in range(len(model_selected)):
        marg = model_selected[i].get_marginal(train_data[0][i])
        marginal.append(np.max(marg, axis=1))  # shape of N*4
    if len(model_selected) > 1:
        sort_comp = ones - np.max(marginal, axis=0)
    else:
        sort_comp = ones - np.max(marginal, axis=1)
    queried_indexs = sorted(range(len(sort_comp)), key=lambda i: sort_comp[i])[-budget:]
    
    return queried_indexs
###############################################################################
