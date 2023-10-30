#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_model_filter_v1: run the script to get the baseline results, the performance of model trained on samples selected
by the common active learning strategies

generates a folder of files that stores the test results inside the experiment folder

"""

__author__      = "Beryl Zhang, Oggi Rudovic"
__copyright__   = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"

### BASE MODEL W DIFFERENT AL STRATEGIES

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import copy

from tagger_v2_average_nobo import Tagger
import load_data_v3_engage as helper
from load_data_v3_engage import num_test_kids, num_train_kids, num_sec
from load_data_v3_engage import test_ids as test_child

from apply_filter_v1 import conservative_sampling, least_confident, diversity_sampling, random_sampling, uncertainty_sampling

from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import csv
from sklearn import preprocessing

# from joblib import Parallel, delayed
# import multiprocessing
# import itertools

AGENT = "LSTMQ"

EXPNUM = 4
NTYPE  = 'd1qn'  # ['d1qn', 'd2qn', 'd3qn', 'd4qn']

NITER = 10

# num_cores = multiprocessing.cpu_count()
# print(num_cores)

# BUDGETS = [5, 10, 20, 50, 200]
BUDGETS = [5]
NSTATES = [0]
# ALMETHODS = ['rs', 'ds', 'lc', 'us']
ALMETHODS = ['rs']

fndict = {"rs": lambda model_selected, budget, train_data: random_sampling(model_selected, budget, train_data),
          "us": lambda model_selected, budget, train_data: uncertainty_sampling(model_selected, budget, train_data),
          "ds": lambda feature_now, model_selected, budget, train_data: diversity_sampling(feature_now, model_selected, budget, train_data), 
          "cs": lambda feature_now, model_selected, budget, train_data: conservative_sampling(feature_now, model_selected, budget, train_data), 
          "lc": lambda feature_now, model_selected, budget, train_data: least_confident(feature_now, model_selected, budget, train_data)}

POLYS = [False]

CUMS = [False]

CONTENTS = [False]

FCLS = False

LOGIT = False

FEATURES =[[0,1,2,3]]#,4,5]]

METHOD = 'maj'

HEADER_WRITTEN = False
################################################################
#feature shape
# FEATURE_SHAPE = [[2048,10],[2048,10],[2048,10],[711,10],[264,10],[264, 10]]
# FEATURE_SHAPE = [[501,10],[501,10],[501,10],[201,10],[201,10],[201,10]]
FEATURE_SHAPE = [[257,10], [70,10], [27,10],[24,10]]
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
    #train_x, train_y, test_x, test_y = helper.test_data(feature_number, student, cvit, which_section)

    train_x, train_y, _ = helper.get_one_data(feature_number, student, which_section)
    test_x, test_y = train_x, train_y

    train = [train_x, train_y]
    test = [test_x, test_y]

    print("Loading environment..")
    # game = Env(story, test, dev, budget, MODEL_VER, model, feature_number, CUM, EXPNUM, cvit, METHOD, num_kids)
    return train, test

def play_ner(feature_now, model_ver_ori, cvits, al_method):
    global BUDGET, test_child, num_sec, HEADER_WRITTEN
    tf.reset_default_graph()        
    
    ################################## construct model(taggers) ###############################

    for cvit in range(cvits):
        tf.reset_default_graph()
        HEADER_WRITTEN = False
        model_ver = model_ver_ori +'_'+str(cvit)

        model_selected = []
        for i in feature_now:
            with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
                model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                               n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],
                               feature_number=i, expnum = EXPNUM, cvit=cvit, loop_i=cvit)
            model_selected.append(model)

        print('Finish model initialization')
            
        #########################################################################################


        ################################## load test child list ###############################
        # test_child = []
        # with open('test.txt') as f:
        #     # load test child names and store as test_child (list)
        #     for line in f.readlines():
        #         l = line.split()[0]
        #         test_child.append(l)
        #########################################################################################


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
            # ######################################################################
            # num_sec_true = min(num_sec, len(sections))
            for i in range(len(model_selected)):
                model_selected[i].student = test_child[ID_student]
            # for which_section in sections[:num_sec_true]:
            which_section = 0
            train, test = initialise_game(model_selected,BUDGET, FEATURE, test_child[ID_student], cvit, which_section)
            train_x_all = copy.deepcopy(train[0])
            train_y_all = copy.deepcopy(train[1])
            if al_method == 'rs':
                queried_indexes = random_sampling(feature_now, model_selected, BUDGET, train, cvit)
            elif al_method == 'ds':
                ############################# reference point ################################
                reference = np.concatenate(train[0], axis=2)
                reference = np.mean(reference, axis=1)
                reference = preprocessing.normalize(reference,axis=0)
                reference=np.mean(reference,axis=0)
                ##############################################################################
                queried_indexes = diversity_sampling(feature_now, model_selected, BUDGET, train, reference.reshape(1,-1))
            else:
                queried_indexes = fndict[al_method](feature_now, model_selected, BUDGET, train)
            
            other_indexs = np.setdiff1d(list(range(len(train[1]))),queried_indexes)

            for i in range(len(model_selected)):
                #################### update to get personalized classifiers ####################
                model_selected[i].train_mode_B(np.array(train[0][i])[queried_indexes], np.array(train[1])[queried_indexes], mode=al_method)
                test[0][i] = np.array(train[0][i][other_indexs])
            test[1] = np.array(train[1])[other_indexs]
            
            write_test_csv(test, model_selected, ID_student, model_ver, cvit, al_method, which_section)
            
            for i in range(len(model_selected)):
                model_selected[i].train_mode_B(np.array(train_x_all[i])[queried_indexes], np.array(train_y_all)[queried_indexes], mode=al_method, type='sections')
                
            print('ID ', ID_student)
            print('> Terminal this child<')

def write_test_csv(test, model, student_ID, model_ver, cvit, al_method, which_section):
    global HEADER_WRITTEN
    """
    write csv records about the results before and after personalization
    :param test_game:               Env, test environment
    :param model:                   list, list of taggers
    :param student_ID:              int, student ID
    :param model_ver:               string, directory to save the recordings
    :param cvit:                    int, test random id
    :param al_method:               string, the active learning method used
    :param which_section:           int, the section id
    :return:
        None
    """

    csv_name = '../../../Filter_base/exp_{0}/test_{1}/results/{2}_'.format(EXPNUM, cvit, al_method)+str(model_ver)+'.csv'
    if not os.path.exists('../../../Filter_base/exp_{0}/test_{1}/results'.format(EXPNUM, cvit) + os.sep):
            os.makedirs('../../../Filter_base/exp_{0}/test_{1}/results'.format(EXPNUM, cvit) + os.sep)

    f = open(csv_name, "a")
    
    writer = csv.DictWriter(
        f, fieldnames=["student_ID",
                       "section",                      
                       "accuracy_majority_test_A",
                       "accuracy_majority_test_{0}".format(al_method),
                       "accuracy_majority_test_section_{0}".format(al_method),
                       "precision_A",
                       "precision_{0}".format(al_method),
                       "precision_section_{0}".format(al_method),
                       "recall_A",
                       "recall_{0}".format(al_method),
                       "recall_section_{0}".format(al_method),
                       "f1_A",
                       "f1_{0}".format(al_method),
                       "f1_section_{0}".format(al_method),
                       "f1_majority_test_A",
                       "f1_majority_test_{0}".format(al_method),
                       "f1_majority_test_section_{0}".format(al_method),
                       "conf_test_A",
                       "conf_test_{0}".format(al_method),
                       "conf_test_section_{0}".format(al_method)
                       ])
    
    if HEADER_WRITTEN == False:
        writer.writeheader()
        HEADER_WRITTEN = True
    
    accuracy_test_A = []
    f1_test_A = []
    conf_test_A = []
    predd_test_A = []
    y_majority_test_A = []
    
    accuracy_test_B = []
    f1_test_B = []
    conf_test_B = []
    predd_test_B = []
    y_majority_test_B = []

    accuracy_test_C = []
    f1_test_C = []
    conf_test_C = []
    predd_test_C = []
    y_majority_test_C = []


    for i in range(len(model)):
        accuracy_test_A.append(float("%.3f" % model[i].test(test[0][i],test[1])))
        f1_test_A.append(float("%.3f" % model[i].get_f1_score(test[0][i],test[1])))

        accuracy_test_B.append(float("%.3f" % model[i].test_B(test[0][i],test[1], mode=al_method)))
        f1_test_B.append(float("%.3f" % model[i].get_f1_score_B(test[0][i],test[1], mode=al_method)))

        accuracy_test_C.append(float("%.3f" % model[i].test_B(test[0][i],test[1], mode=al_method, type='sections')))
        f1_test_C.append(float("%.3f" % model[i].get_f1_score_B(test[0][i],test[1], mode=al_method, type='sections')))
        
        if len(model) ==1:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(test[0][i]))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(test[0][i], mode=al_method))))
            conf_test_C.append(float("%.3f" % np.mean(model[i].get_confidence_B(test[0][i],mode=al_method, type='sections'))))

            predd_test_A.append(model[i].get_predictions(np.squeeze(test[0][i],axis=0)))
            predd_test_B.append(model[i].get_predictions_B(np.squeeze(test[0][i],axis=0), mode=al_method))
            predd_test_C.append(model[i].get_predictions_B(np.squeeze(test[0][i],axis=0), mode=al_method, type='sections'))
        else:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(list(test[0][i])))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(list(test[0][i]), mode=al_method))))
            conf_test_C.append(float("%.3f" % np.mean(model[i].get_confidence_B(list(test[0][i]), mode=al_method, type='sections'))))

            predd_test_A.append(model[i].get_predictions(test[0][i]))
            predd_test_B.append(model[i].get_predictions_B(test[0][i], mode=al_method))
            predd_test_C.append(model[i].get_predictions_B(test[0][i], mode=al_method, type='sections'))

    
    predd_test_A = np.array(predd_test_A).T.tolist()
    predd_test_B = np.array(predd_test_B).T.tolist()
    predd_test_C = np.array(predd_test_C).T.tolist()

    for i in range(len(predd_test_A)):
        preds_test_A = Counter(predd_test_A[i])
        preds_test_B = Counter(predd_test_B[i])
        preds_test_C = Counter(predd_test_C[i])
        y_majority_test_A.append(preds_test_A.most_common(1)[0][0])
        y_majority_test_B.append(preds_test_B.most_common(1)[0][0])
        y_majority_test_C.append(preds_test_C.most_common(1)[0][0])

    results_A = precision_recall_fscore_support(test[1], y_majority_test_A, average=None)
    precision_A, recall_A, f1_A, support_A = results_A
    precision_A = [round(x,3) for x in precision_A] 
    recall_A = [round(x,3) for x in recall_A]
    f1_A = [round(x,3) for x in f1_A]

    results_B = precision_recall_fscore_support(test[1], y_majority_test_B, average=None)
    precision_B, recall_B, f1_B, _ = results_B
    precision_B = [round(x,3) for x in precision_B] 
    recall_B = [round(x,3) for x in recall_B]
    f1_B = [round(x,3) for x in f1_B]

    results_C = precision_recall_fscore_support(test[1], y_majority_test_C, average=None)
    precision_C, recall_C, f1_C, _ = results_C
    precision_C = [round(x,3) for x in precision_C]
    recall_C = [round(x,3) for x in recall_C]
    f1_C = [round(x,3) for x in f1_C]

    f1_majority_test_A = f1_score(test[1], y_majority_test_A, average='macro')
    # f1_majority_test_A = f1_score(test[1], y_majority_test_A, average='weighted')

    accuracy_majority_test_A = sum(np.equal(test[1], y_majority_test_A))/len(test[1])

    f1_majority_test_B = f1_score(test[1], y_majority_test_B, average='macro')
    # f1_majority_test_C = f1_score(test[1], y_majority_test_C, average='weighted')

    accuracy_majority_test_B = sum(np.equal(test[1], y_majority_test_B))/len(test[1])

    f1_majority_test_C = f1_score(test[1], y_majority_test_C, average='macro')
    # f1_majority_test_C = f1_score(test[1], y_majority_test_C, average='weighted')

    accuracy_majority_test_C = sum(np.equal(test[1], y_majority_test_C))/len(test[1])

    writer.writerow({"student_ID": student_ID,
                     "section": which_section,
                     "accuracy_majority_test_A": float("%.3f" % accuracy_majority_test_A),
                     "accuracy_majority_test_{0}".format(al_method): float("%.3f" % accuracy_majority_test_B),
                     "accuracy_majority_test_section_{0}".format(al_method): float("%.3f" % accuracy_majority_test_C),
                     "precision_A": precision_A,
                     "precision_{0}".format(al_method): precision_B,
                     "precision_section_{0}".format(al_method): precision_C,
                     "recall_A": recall_A,
                     "recall_{0}".format(al_method): recall_B,
                     "recall_section_{0}".format(al_method): recall_C,
                     "f1_A": f1_A,
                     "f1_{0}".format(al_method): f1_B,
                     "f1_section_{0}".format(al_method): f1_C,
                     "f1_majority_test_A": float("%.3f" % f1_majority_test_A),
                     "f1_majority_test_{0}".format(al_method): float("%.3f" % f1_majority_test_B),
                     "f1_majority_test_section_{0}".format(al_method): float("%.3f" % f1_majority_test_C),
                     "conf_test_A": conf_test_A,
                     "conf_test_{0}".format(al_method): conf_test_B,
                     "conf_test_section_{0}".format(al_method): conf_test_C
                     })

    print('csv saved')
    f.close()
        
def main(in_iter):
    global AGENT, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, EXPNUM
    

    BUDGET=in_iter[0]
    FEATURE = in_iter[1]
    cvits = in_iter[2]
    al_method = in_iter[3]
    method = 'maj'
    n_states = in_iter[4]
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
            robot = play_ner(FEATURE, MODEL_VER, cvits, al_method)
            tf.reset_default_graph()

                               
 
if __name__ == '__main__':
    # cvits = np.linspace(0,4,5).astype(int)
    cvits = 1
    #### iterate over budget, al_method, cvit
    # Parallel(n_jobs=num_cores)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES,cvits, ALMETHODS,NSTATES))
    main([BUDGETS[0], FEATURES[0], cvits, ALMETHODS[0], NSTATES[0]])
