
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from datetime import datetime


class AttAgent(object):

    def __init__(self, model_file, num_models, expnum, batchsize=32, epochs=10, loop_i=0, n_class=None, save_dir=''):
        self.expnum = expnum
        self.header = save_dir+'results/EXP_{0}/new_checkpoint_attention'.format(self.expnum)
        self.model_file = model_file
        if not os.path.exists(self.header + os.sep + self.model_file):
            os.makedirs(self.header + os.sep + self.model_file)

        self.epochs = epochs
        self.num_models = num_models
        self.batch_size = batchsize
        self.display_step = 30
        self.save_step = 5
        self.n_fea_each = n_class+1
        self.learning_rate = 1e-3
        self.keep_prob = 1.0
        self.mid_states1 = 3*self.num_models
        self.mid_states2 = 2*self.num_models



        def multilayer_perceptron(x, weights, biases, keep_prob):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            layer_2 = tf.nn.dropout(layer_2, keep_prob)

            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        with tf.variable_scope('weight_attention1'):
            self.weights = {'h1': tf.Variable(tf.random_normal([self.n_fea_each*self.num_models, self.mid_states1])),
                            'h2': tf.Variable(tf.random_normal([self.mid_states1, self.mid_states2])),
                            'out': tf.Variable(tf.random_normal([self.mid_states2, self.num_models]))}
        with tf.variable_scope('bias_attention1'):
            self.biases = {'b1': tf.Variable(tf.random_normal([self.mid_states1])),
                           'b2': tf.Variable(tf.random_normal([self.mid_states2])),
                           'out': tf.Variable(tf.random_normal([self.num_models]))}


        self.x = tf.placeholder("float", [None, self.n_fea_each*self.num_models])
        self.y = tf.placeholder("float", [None, self.num_models])


        self.logits = multilayer_perceptron(self.x, self.weights, self.biases, self.keep_prob) # logits

        self.predictions = tf.round(tf.sigmoid(self.logits))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        # pos_weight > 0 will decrease the false negative count, thus increase recall

        self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.logits, pos_weight=1))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.cast(self.y, tf.int64))

        self.correct_pred = tf.equal(self.predictions, self.y)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # tf.summary.scalar('accuracy', self.accuracy)
        
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_att_logs"
        self.step = 0
        logdir = save_dir+"results/EXP_{}/{}_{}/run-{}".format(self.expnum,root_logdir,loop_i,now)

        self.loss_train_summary = tf.summary.scalar('loss_train', self.loss)
        self.acc_train_summary = tf.summary.scalar('acc_train', self.accuracy)
        tf.summary.merge_all()

        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        self.sess = tf.Session(graph=tf.get_default_graph())
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=50)


    def train(self, data_x, data_y, val_x, val_y):
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)        
        if ckpt and ckpt.model_checkpoint_path and (data_x != []):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.initialize_variables(tf.global_variables()))


        # loop over epochs
        if len(data_y) != 0:
            for i in range(self.epochs):
                step = 1
                # in batches
                while (step*self.batch_size) <= len(data_y):
                    batch_x = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                    self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                    summary_str = self.loss_train_summary.eval(session=self.sess,feed_dict={self.x: val_x, self.y: val_y})
                    self.file_writer.add_summary(summary_str, self.step)
                    summary_str = self.acc_train_summary.eval(session=self.sess,feed_dict={self.x: val_x, self.y: val_y})
                    self.file_writer.add_summary(summary_str, self.step)

                    self.step+=1

                    # if step % self.display_step == 0:
                    #     acc = self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    #     loss = self.sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                        #print("Epoch: " + str(i + 1) + ", iter: " + str(
                        #    step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        #    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    step += 1

            # save the model with step of 5
        # if (i+1) % self.save_step == 0:
        

        self.saver.save(self.sess, self.header + os.sep + self.model_file + os.sep + 'model.ckpt', 10)


    def get_predictions(self, x):
        """
        function to get the labels predicted by the current taggers
        :param x:                array, input features
        :return:
            predicted labels
        """
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = self.sess.run(self.predictions, feed_dict={self.x: x})
        return pred






