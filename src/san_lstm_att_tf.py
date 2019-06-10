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

        self._n_dim = options['n_dim']
        self._rnn_size=self._n_dim
        self._batch_size = options['batch_size']
        self._input_embedding_size = options['n_emb']
        self._n_image_feat = options['n_image_feat']
        self._dim_hidden = options['n_common_feat']
        self._dim_att = options['n_attention']
        self._n_words = options['n_words']  
        self._drop_out_rate = options['drop_ratio']
        self._num_output=options['n_output']
        self._n_attention = options['n_attention']
        
        # question-embedding
        self._embed_ques_W = tf.Variable(tf.random_uniform([self._n_image_feat, self._input_embedding_size], -0.08, 0.08), name='embed_ques_W')
        self._w_emb=((np.random.rand(self._n_words, self._input_embedding_size) * 2 - 1) * 0.5).astype(tf.float64)
        
        # encoder: RNN body
        self._lstm_1 = rnn_cell.LSTMCell( self._rnn_size,  use_peepholes=True)
        self._lstm_dropout_1 = rnn_cell.DropoutWrapper(self._lstm_1, output_keep_prob = 1 - self._drop_out_rate)
        self._lstm_2 = rnn_cell.LSTMCell( self._rnn_size, use_peepholes=True)
        self._lstm_dropout_2 = rnn_cell.DropoutWrapper(self._lstm_2, output_keep_prob = 1 - self._drop_out_rate)
        self._stacked_lstm = rnn_cell.MultiRNNCell([self._lstm_dropout_1, self._lstm_dropout_2])

        # image-embedding
        self._embed_image_W = tf.Variable(tf.random_uniform([self._n_image_feat, self._n_dim], -0.08, 0.08), name='embed_image_W')
        self._embed_image_b = tf.Variable(tf.random_uniform([self._n_dim], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self._embed_scor_W = tf.Variable(tf.random_uniform([self._n_dim, self._n_output], -0.08, 0.08), name='embed_scor_W')
        self._embed_scor_b = tf.Variable(tf.random_uniform([self._n_output], -0.08, 0.08), name='embed_scor_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self._n_image_feat, self._n_dim])
        input_idx = tf.placeholder(tf.int32, [self._batch_size, None])
        input_mask = tf.placeholder(tf.int32, [self._batch_size, None])
        label = tf.placeholder(tf.int64, [self._batch_size]) 

        empty_word = np.zeros(1, self._input_embedding_size)
        w_emb_extend = np.concatenate([empty_word,self._w_emb],
                                    axis=0)
        input_emb = w_emb_extend[input_idx]
        input_emb = tf.nn.dropout(input_emb, 1-self._drop_out_rate)
        
        state = self._stacked_lstm.zero_state(self._batch_size, tf.float32)
        loss = 0.0
        
        with tf.variable_scope("embed"):
            for i in range(input_emb.shape[0]):
                output, state = self._stacked_lstm(input_emb, state)

        # multimodal (fusing question & image)
        h_encode = output[-1]

        image_emb = tf.nn.xw_plus_b(image, self._embed_image_W, self._embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            comb_emb = self._attention(h_encode, image_emb)
        with tf.variable_scope("att2"):
            comb_emb = self._attention(comb_emb, image_emb)
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
        return loss, accu
    
    def build_generator(self):
        image = tf.placeholder(tf.float32, [self._n_image_feat, self._n_dim])
        input_idx = tf.placeholder(tf.int32, [self._batch_size, None])
        input_mask = tf.placeholder(tf.int32, [self._batch_size, None])
        label = tf.placeholder(tf.int64, [self._batch_size]) 

        empty_word = np.zeros(1, self._input_embedding_size)
        w_emb_extend = np.concatenate([empty_word,self._w_emb],
                                    axis=0)
        input_emb = w_emb_extend[input_idx]
        input_emb = tf.nn.dropout(input_emb, 1-self._drop_out_rate)
        
        state = self._stacked_lstm.zero_state(self._batch_size, tf.float32)
        loss = 0.0
        
        with tf.variable_scope("embed"):
            for i in range(input_emb.shape[0]):
                output, state = self._stacked_lstm(input_emb, state)

        # multimodal (fusing question & image)
        h_encode = output[-1]

        image_emb = tf.nn.xw_plus_b(image, self._embed_image_W, self._embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            comb_emb = self._attention(h_encode, image_emb)
        with tf.variable_scope("att2"):
            comb_emb = self._attention(comb_emb, image_emb)
        comb_emb = tf.nn.dropout(comb_emb, 1 - self._drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(comb_emb, self._embed_scor_W, self._embed_scor_b) 

        # FINAL ANSWER
        generated_ANS = tf.nn.softmax(scores_emb)

        return generated_ANS

    def attention(self, h_encode, image_emb):
        # Attention weight
        # h_encode-attention
        h_encode_att_W = tf.get_variable('ques_att_W', [self._n_dim, self._n_attention], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        h_encode_att_b = tf.get_variable('ques_att_b', [self._n_attention], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # image-attention
        image_att_W = tf.get_variable('image_att_W', [self._n_dim, self._n_attention], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        image_att_b = tf.get_variable('image_att_b', [self._n_attention], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # probability-attention
        prob_att_W = tf.get_variable('prob_att_W', [self.n_attention, 1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b', [1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

        h_encode_att = tf.tanh(tf.nn.xw_plus_b(h_encode, h_encode_att_W, h_encode_att_b)) # (b x m) x k
        
        image_att = tf.tanh(tf.nn.xw_plus_b(image_emb, image_att_W, image_att_b)) # (b x m) x k

        combined_feat_attention_1 = image_att + \
                                h_encode_att[:, None, :]

        combined_feat_attention_1 = tf.nn.dropout(combined_feat_attention_1, 1 - self._drop_out_rate)

        combined_feat_attention_1 =tf.nn.xw_plus_b(combined_feat_attention_1,prob_att_W,prob_att_b)
        prob_att = tf.nn.softmax(combined_feat_attention_1[:, :, 0])

        image_feat_ave_1 = (prob_att[:, :, None] * image_emb).sum(axis=1)

        comb_emb = tf.add(image_feat_ave_1, h_encode)

        return comb_emb

