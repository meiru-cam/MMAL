# NOTICE: the d2qn, d3qn, d4qn now is unable to use, update required
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.contrib import rnn
import os
import csv
from sklearn.preprocessing import PolynomialFeatures

# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 500  # number of previous transitions to remember
BATCH_SIZE = 100  # size of minibatch
FINAL_EPSILON = 0.005  # the final epsilon
INITIAL_EPSILON = 0.5  # the start initial epsilon
UPDATE_FREQ = 10  # the frequency of updating the tagger
EXPLORE = 100.  # frames over which to anneal epsilon


class RobotLSTMQ:
    def __init__(self, actions, features, content, poly, logit, fcls, ntype, expnum, n_states, loop_i, gamma=GAMMA,
                 n_class=None, save_dir=''):
        """
        initialize the RL agent
        :param actions:             array, the action vector
        :param features:            list, feature type numbers
        :param content:             bool, use content as state or not
        :param poly:                bool, use polynomials of (confidence and marginal)
        :param logit:               array, the logits
        :param fcls:                array, the fcls
        :param ntype:               string, the type of Q-function
        :param expnum:              int, the experiment number
        :param n_states:            int, the number of hidden states of lstm used to squeeze the content features
        :param gamma:               float, the discount factor when calculate Q-value
        :return:
            None
        """
        # replay memory
        self.replay_memory = deque()
        self.time_step = 0.
        self.action = actions
        self.ntype = ntype  # 'd1qn' # ['d1qn', 'd2qn', 'd3qn']
        self.expnum = expnum
        self.episode = 0
        self.loop_i = loop_i
        self.gamma = gamma
        self.save_dir = save_dir
        tf.set_random_seed(loop_i)

        self.train_step = UPDATE_FREQ  # UPDATE FREQUENCY FOR THE Q NETWORK

        self.content = content
        self.logit = logit
        self.poly = poly
        self.fcls = fcls

        # number of features selected
        self.n_features = len(features)
        # shape of all features
        self.features = features
        # self.feature_shape = [[378,10],[257,10],[70,10],[27,10],[24,10]]
        self.epsilon = INITIAL_EPSILON
        self.observe = OBSERVE

        # state dimension after polynomial (only marginals and confidence used)
        # self.expand = [15,45,91,153]

        # number of classes (label class)
        self.num_classes = n_class
        self.n_hidden = n_states
        self.batch_size = BATCH_SIZE
        # create two q networks instead
        self.mainQN = q_network(self, scope='mainQN')

        if self.ntype == 'd2qn' or self.ntype == 'd3qn':  # d1qn, d3qn
            self.targetQN = q_network(self, scope='targetQN')
            self.trainables = tf.trainable_variables()
            self.targetOps = self.updateTargetGraph(tau=0.01)

        self.saver = tf.train.Saver(max_to_keep=20)

        self.sess = tf.Session()
        # ? multiple graphs: how to initialise variables
        self.sess.run(tf.global_variables_initializer())

    ############################# when duoble Q-network ################################
    def updateTargetGraph(self, tau=0.001):
        tfVars = self.trainables
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    def updateTarget(self):
        for op in self.targetOps:
            self.sess.run(op)

    ###################################################################################

    def update_memory(self, confs, preds, actions, rewards, confs2, preds2, terminals):
        """
        update the experience replay memory
        :param confs:              array, confidences, (N,#taggers)
        :param preds:              array, marginal predictives, (N,3*#taggers)
        :param actions:            array, actions (N,1+#features)
        :param rewards:            array, rewards (N,)
        :param confs2:             array, confidences of next states (N,#taggers)
        :param preds2:             array, predictive marginals of next states (N, 3*#taggers)
        :param terminals:          array, boolean array indicate terminate or not
        :return:
            None
        """
        for conf, pred, action, reward, conf2, pred2, terminal in zip(confs, preds, actions, rewards, confs2, preds2,
                                                                      terminals):
            self.replay_memory.append((conf, pred, action, reward, conf2, pred2, terminal))

        while len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

    def update_Q(self, episode, budget, expnum):
        """
        function to update the Q network (specifically d1qn), utilize the train_q_network
        :param episode:             int, the episode number
        :param budget:              int, number of budget seted
        :param expnum:              int, experiment numebr
        :return:
            None
        """

        if len(self.replay_memory) > self.observe:
            # Train the network            
            self.train_qnetwork(budget, expnum)
        # update the episode number
        self.episode = episode

    ############################### update Q-network #########################################
    def train_qnetwork(self, budget, expnum):
        ############################### obtain minibatches #################################
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

        confidence_batch = [data[0] for data in minibatch]
        predictions_batch = [data[1] for data in minibatch]
        action_batch = [data[2] for data in minibatch]
        reward_batch = [data[3] for data in minibatch]
        next_state_confidence_batch = [data[4] for data in minibatch]
        next_state_predictions_batch = [data[5] for data in minibatch]
        ####################################################################################

        for i in range(10):
            ############################### get q_value and action #############################
            y_batch = []
            next_action_batch_main = []
            qvalue_batch_target = []

            # if self.content:
            #     # when content used as state
            #     if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            #         if self.logit:
            #             qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: concatenate_x,
            #                                                                                self.mainQN.xlogits: next_state_logits_batch})
            #         else:
            #             qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
            #                                                 feed_dict={self.mainQN.x: concatenate_x})
            #
            #     elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #         next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.x: concatenate_x,
            #                                                                                self.mainQN.state_confidence: next_state_confidence_batch,
            #                                                                                self.mainQN.predictions: list(
            #                                                                                    concatenate_predictions)})
            #         qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.x: concatenate_x,
            #                                                                              self.targetQN.state_confidence: next_state_confidence_batch,
            #                                                                              self.targetQN.predictions: list(
            #                                                                                  concatenate_predictions)})
            #     else:
            #         print("** Q-learning method not defined.")
            #         raise SystemExit
            #
            # elif self.logit:
            #     # shape of 10*64
            #     # input into the lstm
            #     # there are four, so concatenate to 10*(64*4)
            #     if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            #         qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
            #                                             feed_dict={self.mainQN.xlogits: next_state_logits_batch})
            #     elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #         next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.x: concatenate_x,
            #                                                                                self.mainQN.state_confidence: next_state_confidence_batch,
            #                                                                                self.mainQN.predictions: list(
            #                                                                                    concatenate_predictions)})
            #         qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.x: concatenate_x,
            #                                                                              self.targetQN.state_confidence: next_state_confidence_batch,
            #                                                                              self.targetQN.predictions: list(
            #                                                                                  concatenate_predictions)})
            #     else:
            #         print("** Q-learning method not defined.")
            #         raise SystemExit
            # elif self.fcls:
            #     if self.ntype == 'd1qn':
            #         qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
            #                                             feed_dict={self.mainQN.xfcls: next_state_fcls_batch})
            #     else:
            #         print("** Q-learning method not defined.")
            #         raise SystemExit
            # else:
            #     if self.poly:
            #         enc_total_in = np.concatenate((next_state_confidence_batch, list(concatenate_predictions)), axis=1)
            #         poly = PolynomialFeatures(2)
            #         expanded_enc_total_in = poly.fit_transform(enc_total_in)
            #         if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            #             qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
            #                                                 feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
            #         elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #             next_action_batch_main = self.sess.run(self.mainQN.predict,
            #                                                    feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
            #             qvalue_batch_target = self.sess.run(self.targetQN.qvalue,
            #                                                 feed_dict={self.targetQN.enc_total: expanded_enc_total_in})
            #         else:
            #             print("** Q-learning method not defined.")
            #             raise SystemExit
            #     else:
            #         if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            #             after_bn = self.sess.run(self.mainQN.bn,
            #                                      feed_dict={self.mainQN.state_confidence: next_state_confidence_batch,
            #                                                 self.mainQN.predictions: list(next_state_predictions_batch),
            #                                                 self.mainQN.trainbool: True})
            #             qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
            #                                                 feed_dict={self.mainQN.after_bn: after_bn})
            #
            #         elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #             next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={
            #                 self.mainQN.state_confidence: next_state_confidence_batch,
            #                 self.mainQN.predictions: list(concatenate_predictions)})
            #             qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={
            #                 self.targetQN.state_confidence: next_state_confidence_batch,
            #                 self.targetQN.predictions: list(next_state_predictions_batch)})

            # if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            #     doubleQ = np.argmax(qvalue_batch_target, 1)
            # elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #     doubleQ = qvalue_batch_target[range(self.batch_size), next_action_batch_main]
            # else:
            #     raise NameError('No q-learning defined')

            if self.ntype == 'd1qn':
                after_bn = self.sess.run(self.mainQN.bn,
                                         feed_dict={self.mainQN.state_confidence: next_state_confidence_batch,
                                                     self.mainQN.predictions: list(next_state_predictions_batch),
                                                     self.mainQN.trainbool: True})
                qvalue_batch_target = self.sess.run(self.mainQN.qvalue,
                                                    feed_dict={self.mainQN.after_bn: after_bn})
                doubleQ = np.argmax(qvalue_batch_target, 1)
            else:
                raise NameError('No q-learning defined')

            for i in range(0, BATCH_SIZE):
                terminal = minibatch[i][-1]
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.gamma * doubleQ[i])
            ####################################################################################            

            ##################################### updating #####################################

            # if self.content:
            #     # update the main network
            #     # self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch,
            #     #    self.mainQN.action_input: action_batch, self.mainQN.x: concatenate,
            #     #    self.mainQN.state_confidence: confidence_batch, self.mainQN.predictions:list(concatenate_predictions)})
            #     if self.logit:
            #         self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch,
            #                                                           self.mainQN.action_input: action_batch,
            #                                                           self.mainQN.x: concatenate,
            #                                                           self.mainQN.xlogits: logits_batch})
            #     else:
            #         self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch,
            #                                                           self.mainQN.action_input: action_batch,
            #                                                           self.mainQN.x: concatenate})
            # elif self.logit:
            #     self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch,
            #                                                       self.mainQN.action_input: action_batch,
            #                                                       self.mainQN.xlogits: logits_batch})
            # elif self.fcls:
            #     self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch,
            #                                                       self.mainQN.action_input: action_batch,
            #                                                       self.mainQN.xfcls: fcls_batch})
            # else:
            #     if self.poly:
            #         enc_total_in = np.concatenate((confidence_batch, list(concatenate_predictions)), axis=1)
            #         poly = PolynomialFeatures(2)
            #         expanded_enc_total_in = poly.fit_transform(enc_total_in)
            #         self.sess.run(self.mainQN.updateModel,
            #                       feed_dict={self.mainQN.q_next: y_batch, self.mainQN.action_input: action_batch,
            #                                  self.mainQN.enc_total: expanded_enc_total_in})
            #     else:
            #         after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence_batch,
            #                                                             self.mainQN.predictions: list(
            #                                                                 predictions_batch),
            #                                                             self.mainQN.trainbool: True})
            #         self.sess.run(self.mainQN.updateModel,
            #                       feed_dict={self.mainQN.q_next: y_batch, self.mainQN.action_input: action_batch,
            #                                  self.mainQN.after_bn: after_bn})
            # if self.ntype == 'd2qn' or self.ntype == 'd3qn':
            #     self.updateTarget()

            # update model
            if self.ntype == 'd1qn':
                self.sess.run(self.mainQN.updateModel,
                              feed_dict={self.mainQN.q_next: y_batch, self.mainQN.action_input: action_batch,
                                         self.mainQN.after_bn: after_bn})
            else:
                raise NameError('No q-learning defined')
        ####################################################################################

    # update epsilon with linear greedy
    def change_epsilon(self):
        if self.epsilon > FINAL_EPSILON and len(self.replay_memory) > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    def save_Q_network(self, model_val):
        """
        save the trained Q-network
        :param model_val:               string, the directory of saving the Q-network
        :return:
            None
        """
        npath = self.save_dir + 'results/EXP_{0}/new_checkpoint'.format(self.expnum)
        if not os.path.exists(npath + os.sep + "Q_" + model_val):
            os.makedirs(npath + os.sep + "Q_" + model_val)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

        saver = tf.train.Saver(var_list=bn_moving_vars, max_to_keep=5)
        saver.save(self.sess, npath + os.sep + "Q_" + model_val + os.sep + 'batchvar.ckpt')
        self.saver.save(self.sess, npath + os.sep + "Q_" + model_val + os.sep + 'model.ckpt')

    ### if not meta, action [1,0] -> get label, action [0,1] -> majority voting
    def get_all_action(self, confidence, predictions):
        """
        Get actions of multiple samples at once, used in train mode, with exploration
        specifically when confidence and marginals used as observations

        :param confidence:              array, the confidences of the samples
        :param predictions:             array, the predictive marginals of the samples
        :return:
            actions:                     array, the actions corresonding to the samples
        """

        # confidence in a shape of N*4
        # predictions in a shape of N*12
        # if self.content:
        #     concatenate = sent[-1]
        #     for i in range(self.n_features - 1):
        #         concatenate = np.concatenate((concatenate, sent[i]), axis=1)
        #     # qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate],
        #     if self.logit:
        #         qvalue = self.sess.run(self.mainQN.qvalue,
        #                                feed_dict={self.mainQN.x: [concatenate], self.mainQN.xlogits: [logits]})
        #     else:
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: [concatenate]})
        #
        # elif self.logit:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits: [logits]})
        #
        # elif self.fcls:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls: [fcls]})
        # else:
        #     if self.poly:
        #         enc_total_in = np.concatenate(([confidence], predictions), axis=1)
        #         poly = PolynomialFeatures(2)
        #         expanded_enc_total_in = poly.fit_transform(enc_total_in)
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
        #     else:
        #         after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
        #                                                             self.mainQN.predictions: predictions,
        #                                                             self.mainQN.trainbool: False})
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
                                                            self.mainQN.predictions: predictions,
                                                            self.mainQN.trainbool: False})
        qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        # qvalue : (N, self.action)
        actions = np.zeros((len(confidence), self.action))
        action_indexs = np.argmax(qvalue, axis=1)

        # exploration
        rans = np.random.rand(len(confidence))
        for i, con in enumerate(actions):
            if rans[i] <= self.epsilon:
                # action_query = np.random.randint(2) #either ask or not
                # if action_query == 1:
                #     con[0] = 1
                # else:
                #     action_index = np.random.randint(1,self.action)
                #     con[action_index] = 1

                action_index = np.random.randint(self.action)
                con[action_index] = 1
            else:
                con[action_indexs[i]] = 1
            # actions[i] = con
        return actions

    def get_all_action2(self, confidence, predictions):
        """
        Get actions of multiple samples at once, used in train mode to check the percentage
        that budget can be reached on individual child data, without exploration
        specifically when confidence and marginals used as observations

        :param confidence:              array, the confidences of the samples
        :param predictions:             array, the predictive marginals of the samples
        :return:
            actions:                     array, the actions corresonding to the samples
        """

        # confidence in a shape of N*4
        # predictions in a shape of N*12
        # if self.content:
        #     concatenate = sent[-1]
        #     for i in range(self.n_features - 1):
        #         concatenate = np.concatenate((concatenate, sent[i]), axis=1)
        #     if self.logit:
        #         qvalue = self.sess.run(self.mainQN.qvalue,
        #                                feed_dict={self.mainQN.x: [concatenate], self.mainQN.xlogits: [logits]})
        #     else:
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: [concatenate]})
        #
        # elif self.logit:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits: [logits]})
        #
        # elif self.fcls:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls: [fcls]})
        # else:
        #     if self.poly:
        #         enc_total_in = np.concatenate(([confidence], predictions), axis=1)
        #         poly = PolynomialFeatures(2)
        #         expanded_enc_total_in = poly.fit_transform(enc_total_in)
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
        #     else:
        #         after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
        #                                                             self.mainQN.predictions: predictions,
        #                                                             self.mainQN.trainbool: False})
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
                                                            self.mainQN.predictions: predictions,
                                                            self.mainQN.trainbool: False})
        qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        actions = np.zeros((len(confidence), self.action))
        action_indexs = np.argmax(qvalue, axis=1)
        for i, con in enumerate(actions):
            con[action_indexs[i]] = 1
        return actions

    def test_all_action(self, model_val, confidence, predictions):
        """
        Get actions of multiple samples at once, used in test mode, restore from trained Q
        :param model_val:               string, the directory where the Q-network is saved
        :param observation:             array, the states of the samples
        :param budget:                  int, the budget seted
        :return:
            actions:                    array, the actions corresonding to the samples
            qvalues:                    array, the qvalues of these samples 
        """

        npath = '../../../results/EXP_{0}/new_checkpoint'.format(self.expnum)
        ckpt = tf.train.get_checkpoint_state(npath + os.sep + "Q_" + model_val)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # if self.content:
        #     concatenate = sent[-1]
        #     for i in range(self.n_features - 1):
        #         concatenate = np.concatenate((concatenate, sent[i]), axis=1)
        #     if self.logit:
        #         qvalue = self.sess.run(self.mainQN.qvalue,
        #                                feed_dict={self.mainQN.x: [concatenate], self.mainQN.xlogits: [logits]})
        #     else:
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: [concatenate]})
        #
        # elif self.logit:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits: [logits]})
        #
        # elif self.fcls:
        #     qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls: [fcls]})
        # else:
        #     if self.poly:
        #         enc_total_in = np.concatenate(([confidence], predictions), axis=1)
        #         poly = PolynomialFeatures(2)
        #         expanded_enc_total_in = poly.fit_transform(enc_total_in)
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
        #     else:
        #         after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
        #                                                             self.mainQN.predictions: predictions,
        #                                                             self.mainQN.trainbool: False})
        #         qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        after_bn = self.sess.run(self.mainQN.bn, feed_dict={self.mainQN.state_confidence: confidence,
                                                            self.mainQN.predictions: predictions,
                                                            self.mainQN.trainbool: False})
        qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.after_bn: after_bn})

        actions = np.zeros((len(confidence), self.action))
        action_index = np.argmax(qvalue, axis=1)
        for i, con in enumerate(actions):
            con[action_index[i]] = 1
        return actions, qvalue


################################# Q-network structure #####################################
class q_network():
    def __init__(self, robot, scope):
        self.num_classes = robot.num_classes
        self.n_hidden = robot.n_hidden
        self.n_features = robot.n_features
        self.content = robot.content
        self.features = robot.features
        # self.feature_shape = robot.feature_shape
        self.action = robot.action
        self.poly = robot.poly
        self.fcls = robot.fcls
        # self.expand = robot.expand
        self.ntype = robot.ntype
        self.logit = robot.logit

        ### added this to have consistent inits 
        random.seed(0)

        if self.content:
            def lstm1(x, weights, biases):
                x = tf.unstack(x, 10, 1)
                with tf.variable_scope(scope + '_lstmc1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope + '_lstmc2'):
                    self.outputs1, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs1 = tf.reduce_mean(tf.stack(self.outputs1), 0)

                pred = tf.matmul(self.avg_outputs1, weights['out']) + biases['out']

                return pred

            inputs_all = 0
            for i in self.features:
                inputs_all = self.feature_shape[i][0] + inputs_all

            self.x = tf.placeholder(
                tf.float32, [None, self.feature_shape[i][1], inputs_all], name="input_x")

            # network weights
            # size of a input = 10*3
            self.state_len = inputs_all
            self.weight_proj_c = {'out': tf.Variable(tf.random.normal([self.n_hidden, self.state_len]))}
            self.biases_proj_c = {'out': tf.Variable(tf.random.normal([self.state_len]))}

            # if only content, input to lstm is 10*378
            if self.n_hidden == 0:
                self.enc_s = tf.reduce_mean(tf.stack(self.x), 1)
            else:
                self.enc_s = lstm1(self.x, self.weight_proj_c, self.biases_proj_c)

            if self.logit:
                def lstm2(x, weights, biases):
                    x = tf.unstack(x, 10, 1)
                    with tf.variable_scope(scope + '_lstml1'):
                        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                    with tf.variable_scope(scope + '_lstml2'):
                        self.outputs2, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                    self.avg_outputs2 = tf.reduce_mean(tf.stack(self.outputs2), 0)

                    pred = tf.matmul(self.avg_outputs2, weights['out']) + biases['out']

                    return pred
                self.xlogits = tf.placeholder(tf.float32, [None, 10, self.n_features * 64], name="x_logits")
                self.weight_proj_l = {'out': tf.Variable(tf.random.normal([self.n_hidden, 64 * self.n_features]))}
                self.biases_proj_l = {'out': tf.Variable(tf.random.normal([64 * self.n_features]))}
                self.final_logits = lstm2(self.xlogits, self.weight_proj_l, self.biases_proj_l,
                                          i)  # should have shape of 32
                self.enc_total = tf.concat([self.final_logits, self.enc_s], 1)
            else:
                self.enc_total = self.enc_s
        elif self.logit:
            def lstm2(x, weights, biases):
                x = tf.unstack(x, 10, 1)
                with tf.variable_scope(scope + '_lstml1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope + '_lstml2'):
                    self.outputs2, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs2 = tf.reduce_mean(tf.stack(self.outputs2), 0)
                pred = tf.matmul(self.avg_outputs2, weights['out']) + biases['out']
                return pred

            self.xlogits = tf.placeholder(tf.float32, [None, 10, self.n_features * 64], name="x_logits")
            self.weight_proj_l = {'out': tf.Variable(tf.random.normal([self.n_hidden, 64 * self.n_features]))}
            self.biases_proj_l = {'out': tf.Variable(tf.random.normal([64 * self.n_features]))}

            self.final_logits = lstm2(self.xlogits, self.weight_proj_l, self.biases_proj_l)  # should have shape of 32
            self.enc_total = self.final_logits
        elif self.fcls:
            def lstm2(x, weights, biases):
                x = tf.unstack(x, 10, 1)
                with tf.variable_scope(scope + '_lstmf1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope + '_lstmf2'):
                    self.outputs3, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs3 = tf.reduce_mean(tf.stack(self.outputs3), 0)
                pred = tf.matmul(self.avg_outputs3, weights['out']) + biases['out']
                return pred

            self.xfcls = tf.placeholder(tf.float32, [None, 10, self.n_features * self.num_classes], name="x_logits")
            self.weight_proj_f = {
                'out': tf.Variable(tf.random.normal([self.n_hidden, self.num_classes * self.n_features]))}
            self.biases_proj_f = {'out': tf.Variable(tf.random.normal([self.num_classes * self.n_features]))}

            self.final_xfcls = lstm2(self.xfcls, self.weight_proj_f, self.biases_proj_f)  # should have shape of 32
            self.enc_total = self.final_xfcls
            # only logits, should have a shape of 64*4
        else:
            if self.poly:
                self.enc_total = tf.placeholder(
                    tf.float32, [None, self.expand[self.n_features - 1]], name="input_expanded")
            else:
                self.state_confidence = tf.placeholder(
                    tf.float32, [None, self.n_features], name="input_confidence")
                self.predictions = tf.placeholder(
                    tf.float32, [None, self.n_features * self.num_classes], name="input_predictions")
                self.enc_total = tf.concat((self.state_confidence, self.predictions), axis=1)

        with tf.variable_scope(scope + '_norm_weight'):
            ########## number of units in hidden layer set to 50
            self.w_fc2 = tf.Variable(tf.random.normal([self.enc_total.get_shape().as_list()[-1], 50]))

        with tf.variable_scope(scope + '_norm_bias'):
            self.b_fc2 = tf.Variable(tf.random.normal([50]))
            # if self.ntype == 'd3qn' or self.ntype == 'd4qn':
            #     self.w_fc2 = self.weight_variable([self.enc_total.get_shape().as_list()[-1], 1])

        # bn layer
        # with tf.variable_scope(scope+'_robot'):
        #     # self.w_fc2 = self.weight_variable([self.enc_total.get_shape().as_list()[-1], self.action])
        #     self.w_fc3 = self.weight_variable([50, self.action])
        #     self.b_fc3 = self.bias_variable([self.action])
        #     if self.ntype == 'd3qn' or self.ntype == 'd4qn':
        #         self.w_fc3 = self.weight_variable([self.enc_total.get_shape().as_list()[-1], 1])

        with tf.variable_scope(scope + '_robot_weight'):
            ########## number of units in hidden layer set to 50
            self.w_fc3 = tf.Variable(tf.random.normal([50, self.action]))
        with tf.variable_scope(scope + '_robot_bias'):
            self.b_fc3 = tf.Variable(tf.random.normal([self.action]))

        if self.ntype == 'd3qn' or self.ntype == 'd4qn':
            # Extension to  include Dueling DDQN (Advantage can be negative, that is fine.
            # It means that the action a in A(s,a) is a worse choice than the current policy's)
            # we start with the last layer -- enc_total

            # value function
            self.avalue = tf.matmul(self.enc_total, self.w_fc2) + self.b_fc2
            # advantage
            self.vvalue = tf.matmul(self.enc_total, self.w_fc3)  # + self.b_fc3        # this bias may cause instability
            # Then combine them together to get our final Q-values.
            self.qvalue = self.vvalue + tf.subtract(self.avalue, tf.reduce_mean(self.avalue, axis=1, keep_dims=True))
            # self.qvalue = tf.subtract(self.avalue,tf.reduce_mean(self.avalue,axis=1,keep_dims=True))
        else:
            # Q Value layer
            # pass the batch of data to a normalization layer
            # architecture: bn -> FC -> BN -> FC

            self.trainbool = tf.placeholder(tf.bool, None, 'bn_bool')
            self.bn1 = tf.layers.batch_normalization(self.enc_total, training=self.trainbool)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.bn1 = tf.identity(self.bn1)
            self.enc_total = tf.matmul(self.enc_total,
                                       self.w_fc2) + self.b_fc2  # shape self.enc_total before (N, 24), after (N, 50)
            self.bn = tf.layers.batch_normalization(self.enc_total, training=self.trainbool)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.bn = tf.identity(self.bn)

            # self.after_bn = tf.placeholder(tf.float32, [None, 4*self.n_features], 'feed_enc')
            self.after_bn = tf.placeholder(tf.float32, [None, 50], 'feed_enc')
            self.qvalue = tf.matmul(self.after_bn, self.w_fc3) + self.b_fc3
            ###############################################################################

        self.predict = tf.argmax(self.qvalue, 1)

        # action input
        self.action_input = tf.placeholder("float", [None, self.action])
        self.q_action = tf.reduce_sum(tf.multiply(self.qvalue, self.action_input), reduction_indices=1)

        # reward input
        self.q_next = tf.placeholder("float", [None])

        loss = tf.reduce_sum(tf.square(self.q_next - self.q_action))
        # loss = tf.losses.huber_loss(self.q_next, self.q_action) ### check the error magnitudes + update model function

        # train method
        with tf.variable_scope('adam2'):
            trainer = tf.train.AdamOptimizer(1e-3)
            # trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
            # trainer = tf.train.MomentumOptimizer(learning_rate=1e-2,momentum=0.9, use_nesterov=True)

        self.updateModel = trainer.minimize(loss)

    # def weight_variable(self, shape):
    #     initial = tf.truncated_normal(shape, stddev=0.01)
    #     return tf.Variable(initial)

    # def bias_variable(self, shape):
    #     initial = tf.constant(0.01, shape=shape)
    #     return tf.Variable(initial)
