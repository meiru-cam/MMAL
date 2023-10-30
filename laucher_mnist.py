#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
laucher_v3_new: the main script to run experiment

"""

__author__ = "Beryl Zhang, Oggi Rudovic"
__copyright__ = "Copyright 2020, the laucher for toy example (MNIST) dataset"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Number of GPUs to run on
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter, deque
import multiprocessing

from robotDQN_v2_adam import RobotLSTMQ
from tagger_v2_average_nobo import Tagger
import load_data_v3_engage as helper
from att_agent import AttAgent

from sklearn import preprocessing
from apply_filter_v1 import diversity_sampling, random_sampling, uncertainty_sampling, conservative_sampling, \
    least_confident

##################################################################################################
##################################################################################################
################################### set experiment parameters ####################################

# num_cores = multiprocessing.cpu_count()
# print(num_cores)

AGENT = "LSTMQ"

######################################## create arguments #########################################
expnum_str = 3  # experiment number
data_type = 'mnist'  # which dataset to use, | for the toy example, mnist is used
iter_features = [[0]]  # the selected types of features for experiment | for mnist, there is only 1 feature
iter_dropout = [False]  # include the dropout during training of classifier or not | use the samplest
iter_feature_ori = [False]  # use the original features or teh pca reduced features

# set the maximum number of episode according to the budget
iter_max_episode_dict = [{5: 100, 10: 20, 20: 100, 50: 100, 100: 200}]

iter_ms = [['agent']]  # method to select most informative samples
iter_budget = [5]  # budgets to test, (requires training to be done first)
iter_ntype = ['d1qn']  # the dqn structure, recommend d1qn, the other three didn't updated with the newest architecture
iter_cum = [False]  # whether to accumulate the budget pool or not
iter_nstate = [0]  # number of hidden states in the LSTM in Q-network, not used if content=False
iter_content = [False]  # include the content features to action states or not
iter_fcl = [False]  # include the fcl features to action states or not
iter_logit = [False]  # include the logit to action states or not
iter_poly = [False]  # whether to use polynomial of (confidence and marginals) or not
iter_niter = [10]  # number of epochs in taggers
iter_method = ['maj']  # method to get the final prediction from the predictions of models (different features)


###################################################################################################


##################################################################################################
##################################################################################################
##################################################################################################

def initialise_game(args, num_episodes):
    """
    Function used to initialize the RL environment
    :param args:                    dict, arguments used
    :param num_episodes:            number of episodes, (select samples, train classifier, train RL agent) is one episode
    :return:
    """

    ################################## load data #######################################
    if args.data_type == 'mnist':
        # load toy mnist data, single modality
        train_eval_x, train_eval_y, train_id_list, train_sec_list = helper.load_mnist(args.num_class, 'train',
                                                                                      n_need=3000)
        dev_x_all, dev_y_all, dev_id_list, dev_sec_list = helper.load_mnist(args.num_class, 'dev', n_need=3000)
        test_x_all, test_y_all, test_id_list, test_sec_list = helper.load_mnist(args.num_class, 'test', n_need=3000)
    else:
        # load the dataset for personalisation
        train_eval_x, train_eval_y, train_id_list, train_sec_list = helper.load_all(args.features, 'train',
                                                                                    num_class=args.num_class,
                                                                                    ori=args.feature_ori)
        dev_x_all, dev_y_all, dev_id_list, dev_sec_list = helper.load_all(args.features, 'dev',
                                                                          num_class=args.num_class,
                                                                          ori=args.feature_ori)
        test_x_all, test_y_all, test_id_list, test_sec_list = helper.load_all(args.features, 'test',
                                                                              num_class=args.num_class,
                                                                              ori=args.feature_ori)

    ########## for attention csv writing #########
    for_train_csv = [train_id_list, train_sec_list]
    for_dev_csv = [dev_id_list, dev_sec_list]
    for_test_csv = [test_id_list, test_sec_list]
    #########################################################################################

    ############################### initialise environment ##################################
    dev = [dev_x_all, dev_y_all]
    test = [test_x_all, test_y_all]

    train = []  # train would contain the training data for all episodes
    child = []  # used to save the child ID seelcted for each episode

    if args['data_type'] == 'mnist':
        train_x_all_reorder, train_y_all_reorder = helper.load_mnist_batchs(args.features, type='train', n_need=100)
        print('shape x, ', np.array(train_x_all_reorder).shape, 'shape y', np.array(train_y_all_reorder).shape)
        # shape of x is [1, 467, 100, 1, 784], 1 -> number of feature type, 467 -> number of batches, 100 -> batch_size
        # 1, 784 -> feature size, step: 1, feature dimension: 784

        # shuffle the data
        for i in range(len(train_y_all_reorder)):
            order = np.linspace(0, len(train_y_all_reorder[i]) - 1, len(train_y_all_reorder[i])).astype(int)
            random.shuffle(order)
            train_app = []
            for f in range(len(args['features'])):
                train_app.append(np.array(train_x_all_reorder[f][i])[order])
            train_y_app = np.array(train_y_all_reorder[i])[order]
            train.append([train_app, train_y_app])
    else:
        for episode in range(num_episodes):
            # in the first <args['num_loadall']> episodes, train with the fixed data
            if episode < args['num_loadall']:  # should load for episode 1,2,3,4, the episode 0 is loaded elsewhere
                print('train all loaded')
                train_x_all, train_y_all, _, _ = helper.load_all(args['features'], 'train', num_class=args['num_class'],
                                                                 ori=args['feature_ori'])
            # in the rest episodes, train on individual data (as balance as possible)
            else:
                child_id = random.randint(0, num_train_kids - 1)
                child.append(child_id)
                print('child id loaded', child_id)
                train_x_all, train_y_all = helper.load_trainone(args['features'], child=train_ids[child_id],
                                                                num_class=args['num_class'], ori=args['feature_ori'])

            order = np.linspace(0, len(train_y_all) - 1, len(train_y_all)).astype(int)
            random.shuffle(order)

            train_x_all_reorder = []
            for f in range(len(args['features'])):
                train_x_all_reorder.append(np.array(train_x_all[f])[order])
            train_y_all_reorder = np.array(train_y_all)[order]
            train.append([train_x_all_reorder, train_y_all_reorder])
    #########################################################################################
    return train, dev, test, child, train_eval_x, train_eval_y, [for_train_csv, for_dev_csv, for_test_csv]


def play_ner(args):
    """
    selection samples, update classifier, update the strategy
    :param args:            some of the arguments defined at the start
    :return:
    """
    # dimension of actions
    actions = 2  # dimension of action, 2 means there are two actions [query, predict]
    n_class = args['num_class']
    # max episode differ between budgets to achieve early stop
    max_episode = args['max_episode_dict'][args['budget']]

    random.seed(args['random_seed'])  # set random seed for data loading
    np.random.seed(args['nprandom_seed'])  # set random seed for data loading
    train, dev, test, child, train_eval_x, train_eval_y, for_att_csv = initialise_game(args, max_episode)

    for loop_i in range(args['ite']):
        model_ver = args['model_ver'] + '_' + str(loop_i)
        att_model_ver = 'exp{}_attention_{}'.format(args['expnum'], loop_i)

        tf.reset_default_graph()  # required to repeat the experiment
        if loop_i:
            del robot, model, model_selected, pool_x_buffer, pool_y_buffer

        # set different random seed for the repeated experiment
        random.seed(loop_i)
        np.random.seed(loop_i)

        #################################### initialize RL agent ################################
        if 'agent' in args['ms'] and AGENT == "LSTMQ":
            robot = RobotLSTMQ(actions, args['features'], content=args['content'], poly=args['poly'],
                               logit=args['logit'],
                               fcls=args['fcl'], ntype=args['ntype'], expnum=args['expnum'], n_states=args['nstates'],
                               loop_i=loop_i, n_class=n_class, save_dir=args['save_dir'])
            print('Finish RL_agent initialization')
        else:
            print("** There is no robot, running the baseline")
            # raise SystemExit
        #########################################################################################

        ########################## initialize the attention agent ###############################
        attention_agent = AttAgent(att_model_ver, len(args['features']), args['expnum'], loop_i=loop_i, n_class=n_class,
                                   save_dir=args['save_dir'])
        attention_agent.train([], [], [], [])
        #########################################################################################

        ############################## initialize classifiers ###################################
        model_selected = []  # model for selected features

        acc_old_modalities = []  # save the accuracy of all the modalities after training
        # initialize the classifier
        for i in range(len(args['features'])):
            with tf.name_scope(model_ver + '/feature_{0}'.format(args['features'][i])):
                model = Tagger(model_file=model_ver + '/feature_{0}'.format(args['features'][i]),
                               n_input=args['feature_shape'][args['features'][i]][0],
                               n_steps=args['feature_shape'][args['features'][i]][1],
                               feature_number=args['features'][i], epochs=args['niter'], expnum=args['expnum'],
                               loop_i=loop_i, n_class=n_class,
                               data_type=args['data_type'],
                               save_dir=args['save_dir'])
            model.dropout = args['dropout']  # set dropout
            acc_old_ms = []
            for m in args['ms']:
                acc_old_ms.append(
                    model.train([], [], dev[0][i], dev[1], method=m, acc_old=0, train_eval_x=train_eval_x[i],
                                train_eval_y=train_eval_y))
            model.dropout = False
            model.temp_update(args['ms'][np.argmax(acc_old_ms)])  # update the model with the best selection
            acc_old_modalities.append(np.max(acc_old_ms))
            model_selected.append(model)

        print('Finish classifier initialization')
        #########################################################################################

        #################################### initialize pool ####################################
        # this pool is used to eliminate the small_batch problem during training of classifier
        # the pool will be filled with the already seen data from the agent
        if len(train[0][1]) > args.max_size_pool:
            # fill the pool with training data from episode 1
            tofill_pool = random.sample(range(0, len(train[0][1])), args['max_size_pool'])
        else:
            # fill the pool with training data from episode 1
            tofill_pool = list(range(0, len(train[0][1])))
        pool_x_buffer = []
        for i in range(len(model_selected)):
            pool_x_buffer.append(deque(np.array(train[0][0][i])[tofill_pool]))
        pool_y_buffer = deque(np.array(train[0][1])[tofill_pool])
        #########################################################################################

        path = args['save_dir'] + 'results/EXP_{0}/records/'.format(args['expnum'])
        if not os.path.exists(path):
            os.makedirs(path)
        csvfile = path + str(model_ver) + '.csv'

        episode = 0  # initialize episode
        acc_new = 0  # initialize acc_new, the average accuracy over all the modalities
        flag_csv = False
        while episode < max_episode:
            count_01 = []  # save the distribution of actions

            ############################# randomize the order of samples ########################
            if args.data_type == 'mnist':
                bind = episode % 410  # used when the number of episodes is set to be greater than number of batches
                train_x_all_reorder = train[bind][0]
                train_y_all_reorder = train[bind][1]
            else:
                train_x_all_reorder = train[episode][0]
                train_y_all_reorder = train[episode][1]
            #####################################################################################

            ################################## get all observations #############################
            confidence = np.zeros((len(train_y_all_reorder), len(model_selected)))  # confidence
            pred_class = np.zeros((len(train_y_all_reorder), len(model_selected)))  # predicted labels
            predictions = np.zeros((len(train_y_all_reorder), n_class * len(model_selected)))  # predictive marginals

            for i in range(len(model_selected)):
                confidence[:, i] = model_selected[i].get_confidence(train_x_all_reorder[i])
                predictions[:, i * n_class:i * n_class + n_class] = model_selected[i].get_marginal(
                    train_x_all_reorder[i])
                pred_class[:, i] = model_selected[i].get_predictions(train_x_all_reorder[i])
            #####################################################################################

            ################ get the evaluating sampels for attention agent #####################
            ########### the classifier might be updated thus we need to have the new confidence and marginals
            conf_dev_att = np.zeros((len(dev[1]), len(model_selected)))
            marg_dev_att = np.zeros((len(dev[1]), n_class * len(model_selected)))
            pred_dev_att = np.zeros((len(dev[1]), len(model_selected)))

            for i in range(len(model_selected)):
                conf_dev_att[:, i] = model_selected[i].get_confidence(dev[0][i])
                marg_dev_att[:, i * n_class:i * n_class + n_class] = model_selected[i].get_marginal(dev[0][i])
                pred_dev_att[:, i] = model_selected[i].get_predictions(dev[0][i])

            x_att_dev = np.hstack((marg_dev_att, conf_dev_att))
            y_att_dev = (pred_dev_att == np.reshape(dev[1], (-1, 1))).astype(int)
            #####################################################################################

            ################################# get actions (0s and 1s) ###########################
            if 'agent' in args['ms']:
                actions_all = robot.get_all_action(confidence, predictions)

                index_ones = np.where(np.array(actions_all)[:, 0] == 1)[0]
                index_zeros = np.where(np.array(actions_all)[:, 1] == 1)[0]

                count_01.append(len(index_zeros))
                count_01.append(len(index_ones))
            else:
                index_ones = []
                index_zeros = []
                count_01 = [0, 0]
            #####################################################################################

            ################################## Attention Agent #################################
            x_attention = np.hstack((predictions, confidence))
            y_attention = (pred_class == np.reshape(train_y_all_reorder, (-1, 1))).astype(int)

            if episode > args['start_attention']:
                attention_agent.train(x_attention, y_attention, x_att_dev, y_att_dev)
                pred_att = attention_agent.get_predictions(x_attention)
            else:
                pred_att = np.zeros(shape=(len(train_y_all_reorder), len(model_selected)))

            y_pred = []  # predictions from all modalities
            for i in range(len(model_selected)):
                y_pred.append(model_selected[i].get_predictions(train_x_all_reorder[i]))
            y_pred = np.array(y_pred).T.tolist()  # N x models

            y_majority_pred = []
            for i in range(len(y_pred)):

                ## the general setting with attention agent

                if (pred_att[i] == 0).all():
                    preds = Counter(np.array(y_pred[i]))
                    y_majority_pred.append(preds.most_common(1)[0][0])
                else:
                    preds = Counter(np.array(y_pred[i])[pred_att[i] == 1])
                    y_majority_pred.append(preds.most_common(1)[0][0])

                # if (pred_att[i] == 0.).all():
                #     # if the attention agent result in all zeros, use the majority voting result
                #     preds = Counter(np.array(y_pred[i]))
                #     y_majority_pred.append(preds.most_common(1)[0][0])
                # else:
                #     if train_y_all_reorder[i] in np.array(y_pred[i])[pred_att[i] == 1.]:
                #         # if there is correct prediction in the attention agent selected models
                #         # set the voted result to be the correct label
                #         y_majority_pred.append(train_y_all_reorder[i])
                #     else:
                #         # if the modalities selected by attention agent do not have correct prediction
                #         preds = Counter(np.array(y_pred[i])[pred_att[i] == 1])
                #         try:
                #             y_majority_pred.append(preds.most_common(1)[0][0])
                #         except:
                #             print('except', preds.most_common(1))
                #             y_majority_pred.append(y_pred[i][0])
                #             continue

                ## in this setting, if the correct label is predicted by any of the modalities,
                ## we set the voting result to be the true label

                # if train_y_all_reorder[i] in y_pred[i]:
                #     y_majority_pred.append(train_y_all_reorder[i])
                # else:
                #     preds = Counter(np.array(y_pred[i]))
                #     y_majority_pred.append(preds.most_common(1)[0][0])

            # accuracy when attention agent is used
            acc_att_train = sum((y_majority_pred == train_y_all_reorder)) / len(train_y_all_reorder)
            #####################################################################################

            #################################### compute rewards ################################
            rewards_all = np.zeros(len(train_y_all_reorder))

            if 'agent' in args['ms']:
                rewards_all[index_ones] = -0.5
                rewards_all[index_zeros] = (y_majority_pred == train_y_all_reorder).astype(int)[index_zeros]
                rewards_all[np.where(rewards_all == 0)[0]] = -1
            #####################################################################################

            ################################## set terminal state ###############################
            terminals_all = np.zeros(len(train_y_all_reorder)).astype(bool)
            terminals_all[-1] = True
            #####################################################################################

            train_x_ones = []  # samples asked for annotation
            train_x_zeros = []  # samples do not require annotation, using the predictions
            for i in range(len(model_selected)):
                train_x_ones.append(train_x_all_reorder[i][index_ones])
                train_x_zeros.append(train_x_all_reorder[i][index_zeros])

            ####################### reference point for diversity sampling ######################
            # reference point is train_total because that the features we working on now is model-fusion
            reference = np.concatenate(train_x_all_reorder, axis=2)
            reference = np.mean(reference, axis=1)
            reference = preprocessing.normalize(reference, axis=0)
            reference = np.mean(reference, axis=0)
            #####################################################################################

            # samples randomly selected to be added to the memory of RL for experience replay
            agent_ones = random_sampling(args['features'], args['budget'], [train_x_all_reorder, []])
            agent_zeros = random_sampling(args['features'], args['budget'], [train_x_all_reorder, []])

            ################################## baseline sampling to select ######################
            random_ones = random_sampling(model_selected, args['budget'], [train_x_all_reorder, train_y_all_reorder])
            uncertainty_ones = uncertainty_sampling(model_selected, args['budget'],
                                                    [train_x_all_reorder, train_y_all_reorder])
            # diversity_ones = diversity_sampling(args['features'], model_selected, BUDGET, [train_x_all_reorder, train_y_all_reorder], reference.reshape(1,-1))
            # conservative_ones = conservative_sampling(args['features'], model_selected, BUDGET, [train_x_all_reorder, train_y_all_reorder])
            # lc_ones = least_confident(args['features'], model_selected, BUDGET, [train_x_all_reorder, train_y_all_reorder])
            #####################################################################################

            if 'agent' in args['ms']:
                ################################### diversity sampling ##############################
                # use diversity sampling to control the samples used to train the classifiers
                if len(index_ones) > args['budget'] and len(index_zeros) > args['budget']:
                    select_ones = diversity_sampling(model_selected, args['budget'], [train_x_ones, []],
                                                     reference.reshape(1, -1))
                    train_ones = select_ones
                    select_zeros = diversity_sampling(model_selected, args['budget'], [train_x_zeros, []],
                                                      reference.reshape(1, -1))
                elif len(index_ones) <= args['budget']:
                    select_ones = list(range(len(index_ones)))
                    train_ones = select_ones
                    select_zeros = diversity_sampling(model_selected,
                                                      args['budget'] * 2 - len(select_ones), [train_x_zeros, []],
                                                      reference.reshape(1, -1))
                elif len(index_zeros) <= args['budget']:
                    select_zeros = list(range(len(index_zeros)))
                    train_ones = diversity_sampling(model_selected, args['budget'], [train_x_ones, []],
                                                    reference.reshape(1, -1))
                    select_ones = diversity_sampling(model_selected,
                                                     args['budget'] * 2 - len(select_zeros), [train_x_ones, []],
                                                     reference.reshape(1, -1))
                #####################################################################################

                ################################## update taggers ###################################
                """
                NOTICE: random samples are selected from the pool to be added to the RL_AL selected samples
                        Number of samples use to train the taggers is now (args['num_fill']) in each episode
                """

                #### update the taggers and reboot and then get the next observation of the last instance

                toadd_inx = random.sample(range(0, len(np.array(pool_y_buffer))), args['num_fill'] - len(train_ones))
                toadd_y = np.array(pool_y_buffer)[toadd_inx]

                dict_method = {'agent': index_ones[train_ones], 'rand': random_ones,
                               'uncer': uncertainty_ones}  # , 'diver': diversity_ones, 'conser': conservative_ones, 'lc': lc_ones}
            else:
                dict_method = {'rand': random_ones, 'uncer': uncertainty_ones}
                toadd_inx = random.sample(range(0, len(np.array(pool_y_buffer))), args['num_fill'] - args['budget'])
                toadd_y = np.array(pool_y_buffer)[toadd_inx]

            method_win_modalities = {key: 0 for key in args['ms']}
            acc_all_ms_modal = []
            for i in range(len(model_selected)):
                toadd_x = np.array(pool_x_buffer[i])[toadd_inx]
                # model trained on samples from the buffer and newly selected ones
                model.dropout = args['dropout']
                acc_old_ms = []
                for m in args['ms']:
                    # if episode < 5:
                    acc_old_ms.append(model_selected[i].train(
                        np.concatenate([train_x_all_reorder[i][dict_method[m]], toadd_x]),
                        np.concatenate([train_y_all_reorder[dict_method[m]], toadd_y]), dev[0][i], dev[1], method=m,
                        acc_old=acc_old_modalities, train_eval_x=train_eval_x[i], train_eval_y=train_eval_y))
                    # else:
                    # acc_old_ms[i].append(model_selected[i].train(train_x_all_reorder[i][dict_method[m]],
                    #           train_y_all_reorder[dict_method[m]],dev[0][i], dev[1],  method=m,
                    #           acc_old=acc_old_modalities[i], train_eval_x=train_eval_x[i], train_eval_y=train_eval_y))
                model.dropout = False

                acc_all_ms_modal.append(acc_old_ms)  # (n_modalities, n_ms)
                method_win_modalities[args['ms'][np.argmax(acc_old_ms)]] += 1

            method_win_overall = max(method_win_modalities, key=method_win_modalities.get)
            acc_old_modalities = np.array(acc_all_ms_modal)[:, args['ms'].index(method_win_overall)].tolist()
            # method_win = 'agent'
            if episode == 0:
                acc_new = np.mean(acc_old_modalities)

            if episode == max_episode - 1:
                flag_csv = True

            update, acc_new = helper.write_csv_game2(episode, csvfile, [train_eval_x, train_eval_y], dev, test,
                                                     count_01, rewards_all, model_selected, method_win_overall, acc_new,
                                                     acc_att_train, attention_agent, for_att_csv=for_att_csv,
                                                     expnum=args['expnum'], flag_csv=flag_csv,
                                                     save_dir=args['save_dir'])
            # updating the buffer with the newly selected ones
            for i_pool in dict_method[method_win_overall]:
                pool_x_buffer[i].append(train_x_all_reorder[i][i_pool])
                pool_x_buffer[i].popleft()
                # now y and only ones because y is the same for all
                if i == len(model_selected) - 1:
                    pool_y_buffer.append(train_y_all_reorder[i_pool])
                    pool_y_buffer.popleft()
            #####################################################################################

            if 'agent' in args['ms']:
                ################################### The next states #################################
                confidence2 = np.zeros((len(train_y_all_reorder), len(model_selected)))
                predictions2 = np.zeros((len(train_y_all_reorder), n_class * len(model_selected)))

                confidence2[:-1] = confidence[1:]
                predictions2[:-1] = predictions[1:]

                for i in range(len(model_selected)):
                    confidence2[-1, i] = model_selected[i].get_confidence(train_x_all_reorder[i][0])[0]
                    predictions2[-1, i * n_class:i * n_class + n_class] = model_selected[i].get_marginal(
                        train_x_all_reorder[i][0])

                #####################################################################################
                if update == True:
                    ########################## update experience replay memory ##########################
                    add_replay = select_ones + select_zeros

                    robot.update_memory(confidence[add_replay],
                                        predictions[add_replay],
                                        actions_all[add_replay],
                                        rewards_all[add_replay],
                                        confidence2[add_replay],
                                        predictions2[add_replay],
                                        terminals_all[add_replay])
                    print('update agent')
                else:
                    add_replay = agent_ones + agent_zeros
                    print('the add_replay', add_replay)
                    robot.update_memory(confidence[add_replay],
                                        predictions[add_replay],
                                        actions_all[add_replay],
                                        rewards_all[add_replay],
                                        confidence2[add_replay],
                                        predictions2[add_replay],
                                        terminals_all[add_replay])
                    print('no update agent')
                #####################################################################################

                ############## update Q-function
                robot.update_Q(episode, args['budget'], args['expnum'])
                print('number of samples in experience replay: ', len(robot.replay_memory))

                ############## update epsilon
                robot.change_epsilon()
                robot.save_Q_network(model_ver)

            print("> Episodes finished: ", float("%.3f" % (episode / max_episode)))
            ############## next episode
            episode += 1


def main(args):
    """
    main function, the set experiment with different arguments
    :param in_iter[0]:              int, budget
    :param in_iter[1]:              list, feature type numbers
    :param in_iter[2]:              string, voting method
    :param in_iter[3]:              string, Q-network type
    :param in_iter[4]:              int, number of hidden states for LSTM in the Q-network
    :return:
        None
    """

    # changeable variables
    model_ver_0 = 'exp{8}_model_hidden_{0}_it_{1}_budget_{2}_content_{3}_cum_{4}_logits_{5}_fcls_{6}_{7}'.format(
        args['nstates'], args['niter'], args['budget'], int(args['content']), int(args['cum']), int(args['logit']),
        int(args['fcl']), args['method'], args['expnum'])

    s = [0, 0, 0, 0, 0, 0]

    fvar = '_feature'
    for i in range(np.shape(args['features'])[0]):
        s[args['features'][i]] = 1
    for i in range(np.shape(s)[0]):
        fvar = fvar + '_{0}'.format(s[i])

    model_ver_0 += str(fvar)

    if args['content']:
        args['poly'] = False
        model_ver = model_ver_0 + '_poly_{0}'.format(int(args['poly']))

    else:
        model_ver = model_ver_0 + '_poly_{0}'.format(int(args['poly']))

    args['model_ver'] = model_ver

    tf.reset_default_graph()
    play_ner(args)
    helper.summary_its(args['expnum'], args['save_dir'])


if __name__ == '__main__':
    # Parallel(n_jobs=num_cores-2)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES, METHODS, NTYPES, NSTATES))
    # main([BUDGETS[0], FEATURES[0], METHODS[0], NTYPES[0], NSTATES[0]])

    ###################################################################################################
    options = [[op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14]
               for op0 in iter_features for op1 in iter_dropout for op2 in iter_feature_ori
               for op3 in iter_max_episode_dict for op4 in iter_ms for op5 in iter_budget
               for op6 in iter_ntype for op7 in iter_cum for op8 in iter_nstate
               for op9 in iter_content for op10 in iter_fcl for op11 in iter_logit
               for op12 in iter_poly for op13 in iter_niter for op14 in iter_method]

    max_options = len(options)
    option_counter = 0
    for option in options:
        print('#######################################################################################################')
        print('starting iter ', option_counter, ' out of ', max_options)
        print('option is ', option)
        print('#######################################################################################################')

        args = pd.Series([])
        args['expnum'] = expnum_str + option_counter  # the experiment number
        args['data_type'] = data_type  # which dataset is used
        args['num_class'] = 10  # number of classes for the classification task
        args['num_loadall'] = 5  # for how many episodes, we load data from all instead of current batch
        args['random_seed'] = 123  # seed to control random
        args['nprandom_seed'] = 321  # seed to control np.random
        args['ite'] = 2  # experiment repeat for how many times
        args['max_size_pool'] = 300  # the size of pool to update RL agent
        # args['num_fill'] = 100  # number of samples to train classifier in each episode
        args['start_attention'] = 4  # start to use the attention agent after how many episodes
        args['save_dir'] = 'D:/AL_mnist/'  # the directory to save results
        args['features'] = option[0]  # the selected features
        args['dropout'] = option[1]  # training the classifier with dropout or not
        args['feature_ori'] = option[2]  # use the original data or data after dimension reduction
        args['max_episode_dict'] = option[3]  # adjust the number of episodes based on the budget
        args['ms'] = option[4]  # the sampling method
        args['budget'] = option[5]  # selection budget
        args['num_fill'] = args['budget']  # number of samples
        args['ntype'] = option[6]  # the type of RL algorithm to use, currently just d1qn is able to use
        args['cum'] = option[7]  # whether cumulate the selected samples or not
        args['nstates'] = option[8]  # used when feature content is included in the RL state
        args['content'] = option[9]  # whether include feature content to the RL state or not, currently set to False
        args['fcl'] = option[10]  # whether include feature content to the RL state or not, currently set to False
        args['logit'] = option[11]  # whether include feature content to the RL state or not, currently set to False
        args['poly'] = option[12]  # whether include feature content to the RL state or not, currently set to False
        args['niter'] = option[13]  # number of iterations to train the classifier
        args['method'] = option[14]  # the multimodal voting method

        # args['feature_shape'] = [[501,10],[501,10],[501,10],[201,10],[201,10],[201,10]]
        if args['data_type'] == 'mnist':
            args['feature_shape'] = [[784, 1]]
        elif args['data_type'] == 'tega':
            args['feature_shape'] = [[501, 10], [501, 10], [501, 10], [201, 10], [201, 10], [201, 10]]
            if args['feature_ori'] == True:
                args['feature_shape'] = [[2048, 10], [2048, 10], [2048, 10], [711, 10], [264, 10], [264, 10]]
        else:  # engageme
            args['feature_shape'] = [[257, 10], [70, 10], [27, 10], [24, 10]]

        save_result = args['save_dir']+'results/EXP_' + str(args.expnum)
        if not os.path.exists(save_result):
            os.makedirs(save_result)
        df = pd.DataFrame.from_dict(args)
        df.to_csv(save_result + '/args.txt', header=False, index=True, mode='a')
        main(args)

        option_counter += 1
