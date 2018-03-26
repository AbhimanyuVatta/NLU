#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:32:13 2018

@author: abhimanyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:19:04 2018

@author: abhimanyu
"""

import re
import string
import tensorflow as tf
import numpy as np
import os
import argparse
import datetime as dt
import zipfile
from os import listdir
from os.path import isfile, join

data_path = '/media/abhimanyu/ADI/MTech/CREDIT_SUBJECT_SECONDSEM/NLU/Assignment_2_15251/guttenberg'
data_path_restore = '/media/abhimanyu/ADI/MTech/CREDIT_SUBJECT_SECONDSEM/NLU/Assignment_2_15251/char_logs/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=int, default=3, help="1 - training, 2 - Perplexity, 3 - Generate Sentence")
    opts = parser.parse_args()
    return opts


def parse_text(text):
    # remove the brown POS tags
    text = re.sub("(\/)[^\\ ]+","",text)
    punct = string.punctuation.translate(str.maketrans("", "", ".?!'"))
    for ch in punct:
        text = text.replace(ch, ' ' + ch + ' ')
    for ch in punct:
        text = text.replace(ch, '')
    text = re.sub(r'(?:^| )\w(?:$| )',' ', text)
    text = re.sub("[0-9]", "", text)
    text = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", text)
    text = re.sub("\n(?=[^\n])", " ", text)
    # remove single characters from full text
    text = re.sub(r'(?:^| )\w(?:$| )',' ', text)
    text = text.strip().lower()
    return text


def read_file(filename):
    zip = zipfile.ZipFile(filename)
    zip.extractall(data_path)
    zip.close()
    data_path1 = filename.split('.')[0]+'/'
    files = [f for f in listdir(data_path1) if isfile(join(data_path1, f))]
    data = ''
    for file in files:
        f = open(os.path.join(data_path1, file),encoding="ISO-8859-1")
        text = f.read()
        #data.replace("\n", "<eos>")
        data = data + ' ' + text
        f.close()
    data = parse_text(data)
    return data


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_file(filename)
    
    cnt = set(data)
    #count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    #words, _ = list(zip(*cnt))
    word_to_id = dict(zip(cnt, range(len(cnt))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_file(filename)
    data1 = [word_to_id[word] for word in data if word in word_to_id]
    return data1


def load_data():
    # get the data paths
    #log_dir = '/media/abhimanyu/ADI/MTech/CREDIT_SUBJECT_SECONDSEM/NLU/Assignment2/logs/'
    
    train_path = os.path.join(data_path, "Training.zip")
    valid_path = os.path.join(data_path, "Heldout.zip")
    test_path = os.path.join(data_path, "Testing.zip")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    #print(train_data[:5])
    #print('\n\n WORD-TO-ID\n', word_to_id)
    #print('\n\n VOCABULARY\n', vocabulary)
    #print('\n\n reverse_dict\n', " ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


def batch_producer(raw_data, batch_size, num_steps):
    print('\n\nDATA LEN :', len(raw_data))
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


# create the main model
class Model(object):
    tf.reset_default_graph()
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        # Update the cost
        self.cost = tf.reduce_mean(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.predict_gen = tf.cast(tf.argmax(self.predict, axis=-1), tf.int32)
        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name, model_path,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=200):
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=num_steps, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=num_layers)
    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        check_point = tf.train.get_checkpoint_state(model_path)
        if check_point and check_point.model_checkpoint_path:
            saver.restore(sess, check_point.model_checkpoint_path)
        else:
            init_op = tf.global_variables_initializer()
            sess.run([init_op])
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _,  current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                            step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(sess, data_path_restore + model_save_name)
        # do a final save
        saver.save(sess, data_path_restore + model_save_name)
        # close threads
        coord.request_stop()
        coord.join(threads)


def test(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=batch_size, num_steps=num_steps, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        print(model_path)
        check_point = tf.train.get_checkpoint_state(model_path)
        if check_point and check_point.model_checkpoint_path:
            #print("i am here")
            saver.restore(sess, check_point.model_checkpoint_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                cost, true_vals, pred, current_state, acc = sess.run([m.cost, m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                #true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                #print("True values (1st line) vs predicted values (2nd line):")
                #print(" ".join(true_vals_string))
                print(" ".join(pred_string))
            else:
                cost, acc, current_state = sess.run([m.cost, m.accuracy, m.state], feed_dict={m.init_state: current_state})
        # close threads
        coord.request_stop()
        coord.join(threads)
    
    
def perplexity(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=batch_size, num_steps=num_steps, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        check_point = tf.train.get_checkpoint_state(model_path)
        if check_point and check_point.model_checkpoint_path:
            #print("i am here")
            saver.restore(sess, check_point.model_checkpoint_path)
        # get an average accuracy over num_acc_batches
        tot_cost = 0
        cnt = 0
        for batch in range(test_input.epoch_size):
            cost, acc, current_state = sess.run([m.cost, m.accuracy, m.state], feed_dict={m.init_state: current_state})
            tot_cost += cost
            cnt += 1
        print("Perplexity: {:.3f}".format(np.exp(tot_cost/cnt)))
        # close threads
        coord.request_stop()
        coord.join(threads)


train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
options = parse_args()
num_steps = 10
hidden_size = 36
batch_size = 10
hidden_layers = 10
num_layers = 2
num_epochs = 1

if options.task == 1:
    train(train_data, vocabulary, num_layers=2, num_epochs=num_epochs, batch_size=batch_size, model_save_name='lstm_char', model_path = data_path_restore)

if options.task == 2:
    perplexity(data_path_restore, test_data, reversed_dictionary)

if options.task == 3:
    test(data_path_restore, test_data, reversed_dictionary)