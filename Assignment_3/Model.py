#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:51:30 2018

@author: hemant
"""
import tensorflow as tf
import numpy as np
import os
import sklearn as sk


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    if nlevels == 1:
        max_length = max(map(lambda x : len(list(x)), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


class Model():
    def __init__(self, config):
        self.config = config
        self.idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}
        
    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)
    

    def restore_session(self, dir_model):
        print("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        self.sess.close()


    def add_summary(self):
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.sess.graph)


    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0 
        self.add_summary() 

        for epoch in range(self.config.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
            score = self.run_epoch(train, dev, epoch)

            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                print("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break


    def evaluate(self, test):
        print("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)
    
    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        word_ids, sequence_lengths = pad_sequences(words, 0)
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    

   

    def build(self):
        tf.reset_default_graph()
        
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                print("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdagradOptimizer(self.config.lr)
            self.train_op = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
        return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            print('step:',i+1,' and loss',train_loss)
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)

        return metrics["acc"]

    def run_evaluate(self, test):
            accs = []
            recs=[]
            precs=[]
            f1s = []
            total_true = []
            total_pred =[]
            for words, labels in minibatches(test, self.config.batch_size):
                labels_pred, sequence_lengths = self.predict_batch(words)
                for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                    lab      = lab[:length]
                    lab_pred = lab_pred[:length]
                    total_true.append(lab)
                    total_pred.append(lab_pred)
                    accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                    precs.append(sk.metrics.precision_score(lab, lab_pred, average='micro'))
                    recs.append(sk.metrics.recall_score(lab, lab_pred, average='micro'))
                    f1s.append(sk.metrics.f1_score(lab, lab_pred, average='micro'))
    #                accs.append(sk.metrics.accuracy_score(lab, lab_pred))
            f1 = np.mean(f1s)
            acc = np.mean(accs)
    #        rec = np.mean(recs)
    #        prec =np.mean(precs)
    #        return {"Accuracy": 100*acc, "f1 score": 100*f1, "Precision": 100*prec, "Recall": 100*rec }
            return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds