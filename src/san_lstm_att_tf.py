#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import time
import math
import cv2
import json
import pdb
rnn_cell = tf.contrib.rnn

class Answer_Generator():
    def __init__(self, options):

        self._rnn_size = options['n_dim']
        self._batch_size = options['batch_size']
        self._input_embedding_size = options['n_emb']
        self._dim_image = options['n_image_feat']
        self._dim_hidden = options['n_common_feat']
        self._dim_att = options['n_attention']
        self._n_words = options['n_words']
        self._vocabulary_size = options['n_emb']  
        self._drop_out_rate = options['drop_ratio']
        self._num_output=options['n_output']

        # question-embedding
        self._embed_ques_W = tf.Variable(tf.random_uniform([self._vocabulary_size, self._input_embedding_size], -0.08, 0.08), name='embed_ques_W')

        # encoder: RNN body
        self._lstm_1 = rnn_cell.LSTMCell(self._rnn_size, self._input_embedding_size, use_peepholes=True)
        self._lstm_dropout_1 = rnn_cell.DropoutWrapper(self._lstm_1, output_keep_prob = 1 - self._drop_out_rate)
        self._lstm_2 = rnn_cell.LSTMCell(self._rnn_size, self._rnn_size, use_peepholes=True)
        self._lstm_dropout_2 = rnn_cell.DropoutWrapper(self._lstm_2, output_keep_prob = 1 - self._drop_out_rate)
        self._stacked_lstm = rnn_cell.MultiRNNCell([self._lstm_dropout_1, self._lstm_dropout_2])

        # image-embedding
        self._embed_image_W = tf.Variable(tf.random_uniform([self._dim_image[2], self._dim_hidden], -0.08, 0.08), name='embed_image_W')
        self._embed_image_b = tf.Variable(tf.random_uniform([self._dim_hidden], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self._embed_scor_W = tf.Variable(tf.random_uniform([self._dim_hidden, self._num_output], -0.08, 0.08), name='embed_scor_W')
        self._embed_scor_b = tf.Variable(tf.random_uniform([self._num_output], -0.08, 0.08), name='embed_scor_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self._batch_size, self._dim_image[0], self._dim_image[1], self._dim_image[2]])
        question = tf.placeholder(tf.int32, [self._batch_size, self._max_words_q])
        label = tf.placeholder(tf.int64, [self._batch_size,]) 
        
        state = self._stacked_lstm.zero_state(self._batch_size, tf.float32)
        loss = 0.0
        with tf.variable_scope("embed"):
            for i in range(self._n_words):
                if i==0:
                    ques_emb_linear = tf.zeros([self._batch_size, self._input_embedding_size])
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(self._embed_ques_W, question[:,i-1])

                # LSTM based question model
                ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self._drop_out_rate)
                ques_emb = tf.tanh(ques_emb_drop)

                output, state = self._stacked_lstm(ques_emb, state)

        # multimodal (fusing question & image)
        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self._batch_size, -1])

        image_emb = tf.reshape(image, [-1, self._dim_image[2]]) # (b x m) x d
        image_emb = tf.nn.xw_plus_b(image_emb, self._embed_image_W, self._embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            prob_att1, comb_emb = self._attention(question_emb, image_emb)
        with tf.variable_scope("att2"):
            prob_att2, comb_emb = self._attention(comb_emb, image_emb)
        comb_emb = tf.nn.dropout(comb_emb, 1 - self._drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(comb_emb, self._embed_scor_W, self._embed_scor_b) 

        # Calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=scores_emb)

        # Calculate loss
        loss = tf.reduce_mean(cross_entropy)

        # FINAL ANSWER
        prob = tf.nn.softmax(scores_emb)
        pred_label = tf.argmax(prob, axis=1)
        accu = tf.reduce_mean(np.array_equal(pred_label, label))
        return loss, accu, image, question, label
    
    def build_generator(self):
        image = tf.placeholder(tf.float32, [self._batch_size, self._dim_image[0], self._dim_image[1], self._dim_image[2]])
        question = tf.placeholder(tf.int32, [self._batch_size, self._max_words_q])

        state = self._stacked_lstm.zero_state(self._batch_size, tf.float32)
        with tf.variable_scope("embed"):
            for i in range(self._n_words):
                if i==0:
                    ques_emb_linear = tf.zeros([self._batch_size, self._input_embedding_size])
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(self._embed_ques_W, question[:,i-1])

                ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self._drop_out_rate)
                ques_emb = tf.tanh(ques_emb_drop)
                
                output, state = self._stacked_lstm(ques_emb, state)

        # multimodal (fusing question & image)
        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self._batch_size, -1])

        image_emb = tf.reshape(image, [-1, self._dim_image[2]]) # (b x m) x d
        image_emb = tf.nn.xw_plus_b(image_emb, self._embed_image_W, self._embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            prob_att1, comb_emb = self._attention(question_emb, image_emb)
        with tf.variable_scope("att2"):
            prob_att2, comb_emb = self._attention(comb_emb, image_emb)
        comb_emb = tf.nn.dropout(comb_emb, 1 - self._drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(comb_emb, self._embed_scor_W, self._embed_scor_b) 

        # FINAL ANSWER
        generated_ANS = tf.nn.softmax(scores_emb)

        return generated_ANS, image, question, prob_att1, prob_att2

    def attention(self, question_emb, image_emb):
        # Attention weight
        # question-attention
        ques_att_W = tf.get_variable('ques_att_W', [self._dim_hidden, self._dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        ques_att_b = tf.get_variable('ques_att_b', [self._dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # image-attention
        image_att_W = tf.get_variable('image_att_W', [self._dim_hidden, self._dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        image_att_b = tf.get_variable('image_att_b', [self._dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # probability-attention
        prob_att_W = tf.get_variable('prob_att_W', [self._dim_att, 1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b', [1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

        question_att = tf.expand_dims(question_emb, 1) # b x 1 x d
        question_att = tf.tile(question_att, tf.constant([1, self._dim_image[0] * self._dim_image[1], 1])) # b x m x d
        question_att = tf.reshape(question_att, [-1, self._dim_hidden]) # (b x m) x d
        question_att = tf.tanh(tf.nn.xw_plus_b(question_att, ques_att_W, ques_att_b)) # (b x m) x k
        
        image_att = tf.tanh(tf.nn.xw_plus_b(image_emb, image_att_W, image_att_b)) # (b x m) x k

        output_att = tf.tanh(image_att + question_att) # (b x m) x k
        output_att = tf.nn.dropout(output_att, 1 - self._drop_out_rate)
        prob_att = tf.nn.xw_plus_b(output_att, prob_att_W, prob_att_b) # (b x m) x 1
        prob_att = tf.reshape(prob_att, [self._batch_size, self._dim_image[0] * self._dim_image[1]]) # b x m
        prob_att = tf.nn.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(image_emb, [self._batch_size, self._dim_image[0] * self._dim_image[1], self._dim_hidden]) # b x m x d
        for b in range(self._batch_size):
            image_att.append(tf.matmul(tf.expand_dims(prob_att[b,:],0), image_emb[b,:,:]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.add(image_att, question_emb)

        return prob_att, comb_emb

