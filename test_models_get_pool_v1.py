#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_model_get_pool_v1: run the script after the training of agent, use the trained RL agent to select sample for every
section for every test kid. generate a folder of numpy arraies that stores the indexes of the selected samples

"""

__author__      = "Beryl Zhang, Oggi Rudovic"
__copyright__   = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"


### USE THE TRAINED AGENT TO SELECT SAMPLES

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import numpy as np

from joblib import Parallel, delayed
import multiprocessing
import itertools

import load_data_v3_engage as helper
from load_data_v3_engage import num_test_kids, num_train_kids, num_sec
from load_data_v3_engage import test_ids as test_child


from robotDQN_v2_adam import RobotLSTMQ
from tagger_v2_average_nobo import Tagger


AGENT = "LSTMQ"

##################################################################
# Experiment Number
EXPNUM = 4
META = False

# Q-Network type
NTYPE  = 'd1qn' #'d3qn' #'d1qn', 'd2qn', 'd3qn', 'd4qn']

# Number of epochs in training tagger
NITER = 10

# Parallel setup
num_cores = multiprocessing.cpu_count()

# budgets to test, (requires training to be done first)
# BUDGETS = [5,10,20,50]
BUDGETS = [10]

# number of hidden states in the LSTM in Q-network
# NSTATES = [0, 32, 64, 128]
NSTATES = [0]

# whether to use polynomial of (confidence and marginals) or not
POLYS = [False]

# whether to accumulate the budget pool or not
CUMS = [False] 


# test on only content true, running
# test on only fcls true, running
# test on only logits true, running
# test on only the softmax outputs

CONTENTS = [False]
FCLS = False
LOGIT = False

# if all false: softmax

# feature mode
FEATURES =[[0,1,2,3]]#,4,5]]

# voting methods
METHOD = 'maj'
################################################################
#feature shape
# FEATURE_SHAPE = [[2048,10],[2048,10],[2048,10],[711,10],[264,10],[264, 10]]
# FEATURE_SHAPE = [[501,10],[501,10],[501,10],[201,10],[201,10],[201,10]]
FEATURE_SHAPE = [[257,10], [70,10], [27,10],[24,10]]


#FEATURE = 'ALL','FACE', 'BODY', 'PHY', 'AUDIO']


def initialise_game(model, budget, feature_number, student, cvit, which_section):
    """
    Function used to initialize the RL environment 
    :param model:                list, the taggers created before
    :param budget:               int, the budget for this experiment
    :param feature_number:       list, a list of integers that represent feature types 
    :param student:              string, name of student data file
    :param cvit:                       int, number indicate the test random id
    :return:
        game                     Env, environment for DQN
    """

    ###################################### load data #######################################

    train_x, train_y, _ = helper.get_one_data(feature_number, student, which_section)

    train = [train_x, train_y]


    #######################################################################################
    print("Loading environment..")
    return train

def play_ner(feature_now, model_ver_ori, poly, cvits, logit, fcls, n_states):
    global META, BUDGET, test_child, num_sec
    ################################## construct robot ######################################
    if META:
        n_actions = 1+len(feature_now)
    else:
        n_actions = 2
    n_class = 2

    for cvit in range(cvits):
        tf.reset_default_graph()
        if AGENT == "LSTMQ":
            robot = RobotLSTMQ(n_actions, FEATURE, content = CONTENT, poly = poly, 
                            logit = logit, fcls = fcls, ntype = NTYPE, expnum = EXPNUM, n_states=n_states, loop_i=cvit)
        else:
            print("** There is no robot.")
            raise SystemExit
        #########################################################################################

        ################################## construct model(taggers) ###############################
        model_ver = model_ver_ori +'_'+str(cvit)
        model_selected = []
        for i in feature_now:
            with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
                model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                               n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],
                               feature_number=i, expnum = EXPNUM, cvit=cvit, loop_i=cvit)
            model_selected.append(model)
            
        #########################################################################################


        ################################## load test child list ###############################
        # test_child = []
        # with open('test1.txt') as f:
        #     # load test child names and store as test_child (list)
        #     for line in f.readlines():
        #         l = line.split()[0]
        #         test_child.append(l)
        #########################################################################################

        
        ########################## loop over the test kids #####################################
        for ID_student in range(num_test_kids):
            ######################### get all the sections #####################
            # filenames = []
            # with open('../../../allkids/{}_files.txt'.format(test_child[ID_student])) as f:
            #     for line in f.readlines():
            #         l = line.split()[0]
            #         filenames.append(l)
            # f.close()
            # sections = []
            # [sections.append(filename.split('_')[1]) for filename in filenames]
            # sections = list(set(sections))
            # sections.sort()

            # num_sec_true = min(num_sec, len(sections))

            ######################################################################
            # for which_se ction in sections[:num_sec_true]:
            which_section = 0
            train = initialise_game(model_selected,BUDGET, FEATURE, test_child[ID_student], cvit, which_section)
            confidence = np.zeros((len(train[1]),len(model_selected)))
            predictions = np.zeros((len(train[1]), n_class*len(model_selected)))

            # get confidence and marginal predictives of each classifier
            for i in range(len(model_selected)):
                confidence[:,i]=model_selected[i].get_confidence(train[0][i])
                predictions[:,i*n_class:i*n_class+n_class]=model_selected[i].get_marginal(train[0][i])
            actions, qvalue = robot.test_all_action(model_ver,confidence, predictions)

            # get indexs of all action 1s
            action1s = np.where(actions[:, 0] == 1)[0]

            # store the robot selected samples in the pool
            pool_y = np.array(train[1])[action1s]
            pool_x = []
            for i in range(len(model_selected)):
                pool_x.append(np.array(train[0][i])[action1s])

            # check folder existance
            if not os.path.exists('../../../Filter/exp_{0}/test_{1}'.format(EXPNUM, cvit) + os.sep):
                os.makedirs('../../../Filter/exp_{0}/test_{1}'.format(EXPNUM, cvit) + os.sep)
            
            # save the feature content of samples in pool, loop to save
            # for i in range(len(model_selected)):
                # np.save('Filter_exp_{0}/test_{1}/{5}pool_x{4}_budget_{2}_{3}_sec_{6}.npy'.format(EXPNUM, cvit,BUDGET, test_child[ID_student],i,n_states,which_section),pool_x[i])

            # save the label of samples in pool
            # np.save('Filter_exp_{0}/test_{1}/{4}pool_y_budget_{2}_{3}_sec_{5}.npy'.format(EXPNUM, cvit,BUDGET, test_child[ID_student],n_states, which_section),pool_y)
            # save the qvalues
            np.save('../../../Filter/exp_{0}/test_{1}/{4}budget_{2}_{3}_sec_{5}.npy'.format(EXPNUM, cvit,BUDGET, ID_student,n_states,which_section), qvalue)
            np.save('../../../Filter/exp_{0}/test_{1}/indexs_{4}budget_{2}_{3}_sec_{5}.npy'.format(EXPNUM, cvit,BUDGET, ID_student,n_states,which_section), action1s)

def main(in_iter):
    global AGENT, MAX_EPISODE, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, EXPNUM
    

    BUDGET=in_iter[0]
    FEATURE = in_iter[1]
    cvit = in_iter[2]
    method = 'maj'
    n_states = in_iter[3]
    print(BUDGET, FEATURE)
    for CONTENT in CONTENTS:
        for CUM in CUMS:
            MODEL_VER_0 = 'exp{8}_model_hidden_{0}_it_{1}_budget_{2}_content_{3}_cum_{4}_logits_{5}_fcls_{6}_{7}'.format(n_states, NITER, BUDGET, int(CONTENT), int(CUM), int(LOGIT), int(FCLS), method, EXPNUM)

            s=[0,0,0,0,0,0]

            fvar = '_feature'
            for i in range(np.shape(FEATURE)[0]):
                s[FEATURE[i]]=1
            for i in range(np.shape(s)[0]):
                fvar = fvar+'_{0}'.format(s[i])

            MODEL_VER_0 = MODEL_VER_0 +str(fvar)
            POLY=False
            #The same model_ver
            ####################### test mode A #############################
            MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
            print('test on model ', MODEL_VER)
            robot = play_ner(FEATURE, MODEL_VER, POLY, cvit, LOGIT, FCLS, n_states)
            tf.reset_default_graph()

                               
 
if __name__ == '__main__':
    # cvits = np.linspace(8,9,2).astype(int)
    cvits=5
    main([BUDGETS[0],FEATURES[0],cvits, NSTATES[0]])                           
