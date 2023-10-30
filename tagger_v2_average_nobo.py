#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from sklearn.metrics import f1_score
from datetime import datetime

"""
tagger_v2_average_nobo: the classifier for RL-AL training

functions
    1. def __init__: initialize taggers at the start
    2. def train: train the tagger
    3. def get_predictions: return predicted label
    4. def get_marginals: return predictive marginals
    5. def get_confidence: return confidence
    6. def get_xfcls: return output from the fcls before softmax, shape of (10, 3), in case there are 3 classes
    7. def get_xlogits: return output from the lstm, shape of (10, 64), in case there are 64 lstm hidden states
    8. def get_uncertainty: return uncertainty of model
    
    9. def get_test: return accuracy
    10. def get_f1_score: return the f1_score
    
    10. def train_B: update the tagger with individual test child data, thus generate personalized models
    11. def get_predictions_B: return labels predicted by the personalized model
    12. def get_uncertainty_B: return uncertainty of trained 
"""

__author__ = "Beryl Zhang, Oggi Rudovic"
__copyright__ = "Copyright 2020, Autism Project, extend to test on TEGA data, toy example MNIST"
__version__ = "1.0.1"
__maintainer__ = "Beryl Zhang"
__email__ = "meiru.zhang18@alumni.imperial.ac.uk"


class Tagger(object):
    def __init__(self, model_file, n_steps, n_input, feature_number, training=True, epochs=10, expnum=0, cvit=None,
                 dropout=False, loop_i=0, n_class=None, data_type=None, save_dir=''):
        """
        Initialization function, create taggers (classifiers) with the given parameters
        :param model_file:                string, the directory to save model
        :param n_steps:                   int, the time steps in each sample instance
        :param n_input:                   int, dimension of feature
        :param feature_number:            int, an integer indicate type of feature
        :param training:                  boolean, default True
        :param epochs:                    int, number of epochs during training, default 10
        :param expnum:                    int: the experiment unmber, default 0
        :param cvit:                      int: used in  test mode to distinguish test random id, default None
        :param dropout:                   boolean: include dropout or not, used when compute dropout based active
                                            learning method, default False,
        :param loop_i:                    int: repeat the experiment
        :param n_class:                   int: number of classes
        :param data_type:                 str: which dataset is used
        :return:
            None
        """
        # experiment number
        self.expnum = expnum

        # headers to distinguish train and test
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.logdir = save_dir + "results/EXP_{}/tf_logs_{}/run-{}".format(self.expnum, loop_i, now)
        self.header = save_dir + 'results/EXP_{0}/new_checkpoint'.format(self.expnum)
        self.temp = save_dir + 'results/EXP_{0}/temp_checkpoint'.format(self.expnum)
        self.header_test = save_dir + 'results/EXP_{0}/new_checkpoint_test'.format(self.expnum)
        self.header_section = save_dir + 'results/EXP_{0}/new_checkpoint_test_sections'.format(self.expnum)

        # location to save model
        self.model_file = model_file
        if not os.path.exists(self.header + os.sep + self.model_file):
            os.makedirs(self.header + os.sep + self.model_file)

        if not os.path.exists(self.temp + os.sep + self.model_file):
            os.makedirs(self.temp + os.sep + self.model_file)

        # model used
        self.training = training
        self.learning_rate = 0.001
        self.batch_size = 100

        if data_type == 'mnist':
            self.name = '3-Layer_NN'
        else:
            self.name = "LSTM"
        self.student = ''
        self.dropout = False

        # parameters of LSTM
        self.epochs = epochs
        self.feature_number = feature_number
        self.cvit = cvit
        self.n_input = n_input
        self.n_steps = n_steps
        self.n_hidden = 64
        self.n_classes = n_class
        self.feature_out = 0
        self.inbn = 0  # include bn or not
        tf.set_random_seed(loop_i)

        # base LSTM function
        def lstm(x):
            if self.feature_out:
                print("applying bottleneck: {}".format(self.feature_out))
                with tf.variable_scope('weights_bottleneck_f{0}'.format(self.feature_number)):
                    self.weights_bottleneck = {'bottleneck': tf.Variable(tf.random.normal([self.n_input,
                                                                                           self.feature_out]))}
                with tf.variable_scope('bias_bottleneck_f{0}'.format(self.feature_number)):
                    self.bias_bottleneck = {'bottleneck': tf.Variable(tf.random.normal([self.feature_out]))}
                x = tf.map_fn(lambda x_e: tf.matmul(x_e, self.weights_bottleneck['bottleneck']) +
                                          self.bias_bottleneck['bottleneck'], x, dtype=tf.float32)

                # for j in range(self.n_steps):
                #     x = tf.matmul(x, self.weights_bottleneck['bottleneck'])+ self.bias_bottleneck['bottleneck']

                x = tf.convert_to_tensor(x)  # N x 10 x DIM --> N*10 x DIM --> apply BN --> reshape to N x 10 x DIM

            if self.inbn:
                x = tf.reshape(x, (-1, self.n_input))
                self.bn_tagger = tf.layers.batch_normalization(x, training=self.is_training,
                                                               name='bn_tagger_f{}'.format(self.feature_number))

                all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                update_ops = []
                for op_i in all_update_ops:
                    if 'bn_tagger_f{}'.format(self.feature_number) in op_i.name:
                        update_ops.append(op_i.name)
                # print('tf.graph op in tagger', tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                with tf.control_dependencies(tf.get_collection(tf.convert_to_tensor(update_ops))):
                    self.bn_tagger = tf.identity(self.bn_tagger)
                    # self.avg_outputs = tf.layers.batch_normalization(self.avg_outputs, training=True,
                    #                                           name='bn_tagger_{}'.format(self.feature_number))

                # self.after_bn = tf.placeholder(tf.float32, [self.batch_size*self.n_steps, self.n_input],
                #                                               'avg_outputs_after_bn')
                x = tf.reshape(self.bn_tagger, (-1, self.n_steps, self.n_input))
            else:
                print('nobn')
            x = tf.unstack(x, self.n_steps, 1)

            # by default not included, use by adding dropout=True when initialize model
            # if dropout:
            #     keep_prob_in = 0.75
            #     x = [tf.nn.dropout(x_i, keep_prob_in) for x_i in x]

            ##################################################################################
            with tf.variable_scope('tagger1_feature_f{0}'.format(self.feature_number)):
                lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                # lstm_cell = rnn.LayerNormBasicLSTMCell(self.n_hidden, forget_bias=1.0)
            with tf.variable_scope('tagger2_feature_f{0}'.format(self.feature_number)):
                self.outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                # take the self.outputs to the Q-network

            # if dropout:
            #     keep_prob_out = 0.5
            #     self.outputs = [tf.nn.dropout(rnn_output, keep_prob_out) for rnn_output in self.outputs]
            ###################################################################################
            logitx = tf.stack(self.outputs, axis=1)  # shape of 10*64

            # matmul = tf.matmul(logitx, weights['out']) + biases['out']
            # self.avg_outputs = tf.reduce_mean(tf.stack(self.outputs), 0) # (10,?,64) -> (?, 64)
            # self.avg_outputs = self.outputs[-1]
            avg_outputs_before_bn = tf.reduce_mean(tf.stack(self.outputs), 0)

            # not used
            matmul = tf.zeros([10, 3], tf.int32)
            return avg_outputs_before_bn, logitx, matmul

        def weight_variable(name, shape):
            """
            Create a weight variable with appropriate initialization
            :param name: weight name
            :param shape: weight shape
            :return: initialized weight variable
            """
            initer = tf.truncated_normal_initializer(stddev=0.01)
            return tf.get_variable('W_' + name,
                                   dtype=tf.float32,
                                   shape=shape,
                                   initializer=initer)

        def bias_variable(name, shape):
            """
            Create a bias variable with appropriate initialization
            :param name: bias variable name
            :param shape: bias variable shape
            :return: initialized bias variable
            """
            initial = tf.constant(0., shape=shape, dtype=tf.float32)
            return tf.get_variable('b_' + name,
                                   dtype=tf.float32,
                                   initializer=initial)

        def fc_layer(x, num_units, name, use_relu=True):
            """
            Create a fully-connected layer
            :param x: input from previous layer
            :param num_units: number of hidden units in the fully-connected layer
            :param name: layer name
            :param use_relu: boolean to add ReLU non-linearity (or not)
            :return: The output array
            """
            indim = x.get_shape()[-1]
            W = weight_variable(name, shape=[indim, num_units])
            b = bias_variable(name, [num_units])
            layer = tf.matmul(x, W)
            layer += b
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

        def nn(x):
            x = tf.reshape(x, shape=(-1, self.n_input))
            fc1 = fc_layer(x, 200, 'FC1', use_relu=True)
            output_logits = fc_layer(fc1, self.n_classes, 'OUT', use_relu=False)
            return output_logits

        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("int32", [None])
        self.is_training = tf.placeholder(tf.bool, None, 'bool_bn_f{}'.format(self.feature_number))

        # Define weights
        if data_type != 'mnist':
            initer = tf.truncated_normal_initializer(stddev=0.01)
            self.weights = tf.get_variable('weight_feature_f{0}_out'.format(self.feature_number),
                                           dtype=tf.float32,
                                           shape=[self.n_hidden, self.n_classes],
                                           initializer=initer)
            # with tf.variable_scope('weight_feature_f{0}'.format(self.feature_number)):
            #      self.weights = {'out': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes]))}
            with tf.variable_scope('bias_feature_f{0}'.format(self.feature_number)):
                self.biases = {'out': tf.Variable(tf.random.normal([self.n_classes]))}

            self.avg_outputs, self.xlogits, self.xfcls = lstm(self.x)
            self.avg_outputs = tf.nn.relu(self.avg_outputs)

            if self.dropout:
                self.avg_outputs = tf.nn.dropout(self.avg_outputs, keep_prob=0.75)
            self.pred = tf.matmul(self.avg_outputs, self.weights) + self.biases['out']
        else:
            self.pred = nn(self.x)

        self.sm = tf.nn.softmax(self.pred)
        # self.y is one-hot vector therefore use the sparse_softmax_cross_entropy
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        # all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # update_ops = []
        # for op_i in all_update_ops:
        #     if '_f{}'.format(self.feature_number) in op_i.name:
        #         update_ops.append(op_i.name)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        with tf.variable_scope('adam1_feature_f{0}'.format(self.feature_number)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9, use_nesterov=True).minimize(self.loss)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.sm, 1), tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.best_cost = 0

        # record to tensorflow board
        self.loss_train_summary = tf.summary.scalar('loss_train', self.loss)
        self.loss_val_summary = tf.summary.scalar('loss_val', self.loss)
        tf.summary.merge_all()

        self.step = 0
        self.file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

        # self.sess = tf.Session(graph=tf.get_default_graph(),config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session(graph=tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=50)

    def train(self, data_x, data_y, val_x, val_y, method='agent', acc_old=None, train_eval_x=None, train_eval_y=None):
        """
        :param data_x:                      array, features of the training data
        :param data_y:                      array, labels of the training data
        :param val_x:                       array, features of the validation data
        :param val_y:                       array, labels of the validation data
        :param method:                      str, which sampling method is used to select the training data
        :param acc_old:                     float, the accuracy of the last episode, used for early stopping
        :param train_eval_x:                array, samples from training data but not the same as the data_x
        :param train_eval_y:                array, labels of train_eval_y
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        if ckpt and ckpt.model_checkpoint_path and (data_x != []):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.initialize_variables(tf.global_variables()))

        if not os.path.exists(self.temp + os.sep + self.model_file + os.sep + method + os.sep):
            os.makedirs(self.temp + os.sep + self.model_file + os.sep + method + os.sep)

        # When the number of samples given to train tagger is large enough, set batch_size to 32
        # if len(data_y) >= 50:
        #     self.batch_size = 32

        # loop over epochs
        # acc_old = 0
        # acc_new = 0
        if len(data_y) == 0:
            acc_new = self.sess.run(self.accuracy, feed_dict={self.x: val_x, self.y: val_y})
        count_n = 0
        if len(data_y) > 0:
            last_improvement = 0
            for i in range(self.epochs):
                step = 1
                # in batches
                # while step * self.batch_size <= len(data_y):
                # batch_x = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                # batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                batch_x = data_x
                batch_y = data_y

                summary_str = self.loss_train_summary.eval(session=self.sess,
                                                           feed_dict={self.x: train_eval_x, self.y: train_eval_y,
                                                                      self.is_training: False})
                self.file_writer.add_summary(summary_str, self.step)

                summary_str = self.loss_val_summary.eval(session=self.sess, feed_dict={self.x: val_x, self.y: val_y,
                                                                                       self.is_training: False})
                self.file_writer.add_summary(summary_str, self.step)
                self.step += 1

                self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                acc_new, cur_loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.x: val_x, self.y: val_y})

                # avg_cost = acc_new #self.sess.run(self.loss, feed_dict={self.x: val_x, self.y: val_y, self.is_training: False})
                # if avg_cost >= self.best_cost:
                #     self.best_cost = avg_cost
                #     last_improvement = 0
                #     self.saver.save(self.sess, self.temp + os.sep + self.model_file + os.sep + method + os.sep +'model.ckpt', global_step=10)
                #     # self.saver.save(self.sess, self.header + os.sep + self.model_file + os.sep + 'model.ckpt', self.epochs)
                # else:
                #     last_improvement +=1

                # if last_improvement == 4:
                #     print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                #     break

                self.saver.save(self.sess,
                                self.temp + os.sep + self.model_file + os.sep + method + os.sep + 'model.ckpt',
                                global_step=10)
                ## early stoping
                # if acc_new >= acc_old:
                #     count_n = 0 # when the acc_new is smaller for continuously 3 epochs, stop                   
                #     acc_old = acc_new
                #     continue
                # else:
                #     count_n += 1
                #     if count_n == 4:
                #         print('stop at epoch ', i)
                #         break
                ##
                step += 1
        else:
            self.saver.save(self.sess, self.temp + os.sep + self.model_file + os.sep + method + os.sep + 'model.ckpt',
                            global_step=10)
        return acc_new

    def temp_update(self, method='agent'):
        """
        update the weights and parameters from temporal directory to the true directory
        :param method:              which model to be updated with
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(self.temp + os.sep + self.model_file + os.sep + method)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.saver.save(self.sess, self.header + os.sep + self.model_file + os.sep + 'model.ckpt', self.epochs)

    def get_predictions_temp(self, x, method='agent'):
        """
        get predictions based on the checkpoint saved in temporal dir
        :param x:                   array, input data to get prediction
        :param method:              str, which model to use
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(self.temp + os.sep + self.model_file + os.sep + method)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.sm, feed_dict={self.x: x}), 1)
        return pred

    def get_predictions(self, x):
        """
        function to get the labels predicted by the current taggers
        :param x:                array, input features
        :return:
            predicted labels
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.sm, feed_dict={self.x: x}), 1)
        return pred

    def get_marginal(self, x):
        """
        function to get the predictive marginals
        :param x:                array, input features
        :return:
            predictive marginals
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(x).shape) == 2:
            marginal = self.sess.run(self.sm, feed_dict={self.x: [x]})
        else:
            marginal = self.sess.run(self.sm, feed_dict={self.x: x})
        return marginal

    def get_confidence(self, x):
        """
        function to get the confidence of prediction by the current taggers
        :param x:                array, input features
        :return:
            confidence of each sample instance
        """

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # if multiple sampels
        if len(np.array(x).shape) == 3 and len(x) > 1:
            margs = self.sess.run(self.sm, feed_dict={self.x: x})
            margs = margs + np.finfo(float).eps
            margs = -np.sum(np.multiply(margs, np.log(margs)), axis=1)
            margs = np.minimum(1, margs)
            margs = np.maximum(0, margs)
            # conf  = np.mean(1-margs)
            conf = 1 - margs
        else:
            margs = self.sess.run(self.sm, feed_dict={self.x: [x]})
            conf = [1 - np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs + np.finfo(float).eps))))]

        return conf

    def get_xfcls(self, x):
        """
        function to get the output from the fcls before softmax, shape of (10, 3), in case there are 3 classes
        :param x:                array, input features
        :return:
            output from fcLs before average
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        if len(np.array(x).shape) == 2:
            xfcls = self.sess.run(self.xfcls, feed_dict={self.x: [x]})
        else:
            xfcls = self.sess.run(self.xfcls, feed_dict={self.x: x})
        return xfcls

    def get_xlogits(self, x, y):
        """
        function to get the output from the lstm, shape of (10, 64), in case there are 64 lstm hidden states
        :param x:                array, input features
        :param y:                array, true labels
        :return:
            output from lstm states
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x).shape) == 2:
            logits = self.sess.run(self.xlogits, feed_dict={self.x: [x], self.y: [y]})
        else:
            logits = self.sess.run(self.xlogits, feed_dict={self.x: x, self.y: y})
        return logits

    def get_uncertainty(self, x, y):
        """
        function to get the uncertainty of each prediction, here is cross entropy loss
        :param x:                array, input features
        :param y:                array, true labels
        :return:
            uncertainty
        """

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x).shape) == 2:
            loss = self.sess.run(self.loss, feed_dict={self.x: [x], self.y: y})
        else:
            loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y: y})
        return loss

    def test(self, X_test, Y_true):
        """
        function to get the accuracy of each prediction, here is cross entropy loss
        :param X_test:                array, input features
        :param Y_true:                array, true labels
        :return:
            accuracy of the predicitons (#correct predicts/#samples)
        """

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        # var_list = [v for v in tf.global_variables() if 'weights_bottleneck' in v.name]
        # wei_to_see=self.sess.run(var_list[3*self.feature_number])
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        acc = self.sess.run(self.accuracy, feed_dict={self.x: X_test, self.y: Y_true})
        # f_1 score and conf matrix
        return acc

    def get_f1_score(self, Y_true, X_test):
        """
        function to get f1 score of each prediction, here is cross entropy loss
        :param X_test:                array, input features
        :param Y_true:                array, true labels
        :return:
            f1 score
        """

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: X_test}), 1)

        # f1 score here is the averaged f1 score
        # f1 = f1_score(Y_true, pred, average='macro')

        # could be changed to weighted f1 score, might be better for imbalanced data
        f1 = f1_score(Y_true, pred, average='weighted')

        return f1

    def train_mode_B(self, data_x, data_y, mode='B', type=''):
        """
        restore the parameters from the model trained in 'laucher'    
        if not exist, there is an error

        :param data_x:                array, the training input features
        :param data_y:                array, the training labels
        :param mode:                  string, the AL method used to select training samples
        :param type:                  string, update based on the result from previous section or from base
        :return:
            None
        """

        ############################# select the proper directory to load model from ###############################
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                os.makedirs(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                      self.student) + self.model_file)
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            if not os.path.exists(
                    self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file):
                os.makedirs(self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)
            ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        ################################################################################################################

        ################################################ training ######################################################
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('###############################ERROR#############################')

        # if len(data_y) >= 50:
        #     self.batch_size = 32

        for i in range(self.epochs):
            step = 1
            if len(data_y) != 0:
                # while step * self.batch_size <= len(data_y):

                # batch_x = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                # batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]
                batch_x = data_x
                batch_y = data_y

                self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                step += 1
        # save the updated model in corresponding folder
        # if (i+1) % self.save_step == 0:
        if type == 'sections':
            self.saver.save(self.sess, self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                            self.student) + self.model_file + os.sep + 'model.ckpt', self.epochs)
        elif type == 'pool_sections':
            self.saver.save(self.sess, self.header_section + os.sep + 'pool_test_{0}_{1}/{2}/'.format(mode, self.cvit,
                            self.student) + self.model_file + os.sep + 'model.ckpt', self.epochs)
        else:
            self.saver.save(self.sess, self.header_test + os.sep + 'test_{0}_{1}/'.format(mode,
                            self.cvit) + self.model_file + os.sep + 'model.ckpt', self.epochs)

    """
    The following function have the same usage as the functions above but are using the personalized models
    Mode here indicate which model to load based on the AL method use to select samples
    """
    def get_predictions_B(self, x, mode='B', type=''):
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            ckpt = tf.train.get_checkpoint_state(
                self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)

        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one
        pred = np.argmax(self.sess.run(self.sm, feed_dict={self.x: x}), 1)
        return pred

    def test_B(self, X_test, Y_true, mode='B', type=''):
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            ckpt = tf.train.get_checkpoint_state(
                self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)

        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        acc = self.sess.run(self.accuracy, feed_dict={self.x: X_test, self.y: Y_true})
        # f_1 score and conf matrix
        return acc

    def get_f1_score_B(self, X_test, Y_true, mode='B', type=''):
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            ckpt = tf.train.get_checkpoint_state(
                self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)

        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: X_test}), 1)

        # f1 = f1_score(Y_true, pred, average='macro')

        f1 = f1_score(Y_true, pred, average='weighted')
        return f1

    def get_confidence_B(self, x, mode='B', type=''):
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            ckpt = tf.train.get_checkpoint_state(
                self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x).shape) == 3 and len(x) > 1:

            margs = self.sess.run(self.sm, feed_dict={self.x: x})
            margs = margs + np.finfo(float).eps
            margs = -np.sum(np.multiply(margs, np.log(margs)), axis=1)
            margs = np.minimum(1, margs)
            margs = np.maximum(0, margs)
            conf = 1 - margs
            # conf  = np.mean(1-margs)

        else:
            margs = self.sess.run(self.sm, feed_dict={self.x: [x]})
            conf = [1 - np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs + np.finfo(float).eps))))]

        return conf

    def get_uncertainty_B(self, x, y, mode='B', type=''):
        if type == 'sections':
            if not os.path.exists(self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                                    self.student) + self.model_file):
                ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
            else:
                ckpt = tf.train.get_checkpoint_state(
                    self.header_section + os.sep + 'test_{0}_{1}/{2}/'.format(mode, self.cvit,
                                                                              self.student) + self.model_file)
        else:
            ckpt = tf.train.get_checkpoint_state(
                self.header_test + os.sep + 'test_{0}_{1}/'.format(mode, self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        loss = self.sess.run(self.loss, feed_dict={self.x: [x], self.y: y})
        return loss
