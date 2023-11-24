import numpy as np
import os
import random
import re
import time

from base.base_model import BaseModel
import tensorflow as tf


class HATS(BaseModel):
    def __init__(self, config):
        super(HATS, self).__init__(config)
        self.n_labels = len(config.label_proportion)
        self.input_dim = len(config.feature_list)
        self.num_layer = config.num_layer
        self.keep_prob = 1-config.dropout
        self.max_grad_norm = config.grad_max_norm
        self.num_relations = config.num_relations
        self.node_feat_size = config.node_feat_size
        self.rel_projection = config.rel_projection
        self.feat_attention = config.feat_att
        self.rel_attention = config.rel_att
        self.att_topk = config.att_topk
        self.num_companies = config.num_companies
        self.neighbors_sample = config.neighbors_sample


        self.build_model()
        self.init_saver()

    def get_state(self, state_module):
        if state_module == 'lstm':
            cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.node_feat_size) for _ in range(1)]
            dropout = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob,
                                        output_keep_prob=self.keep_prob) for cell in cells]
            lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(dropout, state_is_tuple=True)
            outputs, state = tf.compat.v1.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.compat.v1.float32)
            state = tf.compat.v1.concat([tf.compat.v1.zeros([1,state[-1][-1].shape[1]]), state[-1][-1]], 0) # zero padding

        return state

    def relation_projection(self, state, rel_idx):
        with tf.compat.v1.variable_scope('relation_'+str(rel_idx)):
            rel_state = tf.compat.v1.layers.dense(inputs=state, units=self.node_feat_size,
                                activation=tf.compat.v1.nn.leaky_relu, name='projection')
        return rel_state

    def node_level_attention(self, node_feats, rel_mat, rel_num, rel_idx):
        with tf.compat.v1.variable_scope('node_attention_'+str(rel_idx)):
            # Neighbors [N, max_neighbors, Node feat dim]
            neighbors = tf.compat.v1.nn.embedding_lookup(node_feats, rel_mat)
            mask_mat = tf.compat.v1.to_float(tf.compat.v1.expand_dims(tf.compat.v1.sequence_mask(rel_num, self.max_k), -1))
            exp_node_feats = tf.compat.v1.tile(tf.compat.v1.expand_dims(node_feats[1:], 1), [1, self.max_k, 1])
            att_x = tf.compat.v1.concat([neighbors, exp_node_feats], -1)
        return att_x

    def get_relation_rep(self, state):
        # input state [Node, Original Feat Dims]
        with tf.compat.v1.variable_scope('graph_ops'):
            if self.feat_attention:
                neighbors = tf.compat.v1.nn.embedding_lookup(state, self.rel_mat)
                # exp_state [1, Nodes, 1, Feat dims]
                exp_state = tf.compat.v1.expand_dims(tf.compat.v1.expand_dims(state[1:], 1), 0)
                exp_state = tf.compat.v1.tile(exp_state, [self.num_relations, 1, self.max_k, 1])
                rel_embs = self.to_input_shape(self.rel_emb)

                # Concatenated (Neightbors with state) :  [Num Relations, Nodes, Num Max Neighbors, 2*Feat Dims]
                att_x = tf.compat.v1.concat([neighbors, exp_state, rel_embs], -1)

                score = tf.compat.v1.layers.dense(inputs=att_x, units=1, name='state_attention')
                att_mask_mat = tf.compat.v1.to_float(tf.compat.v1.expand_dims(tf.compat.v1.sequence_mask(self.rel_num, self.max_k), -1))
                att_score = tf.compat.v1.nn.softmax(score, 2)
                all_rel_rep = tf.compat.v1.reduce_sum(neighbors*att_score, 2) / tf.compat.v1.expand_dims((tf.compat.v1.to_float(self.rel_num)+1e-10), -1)

        return all_rel_rep

    def get_relations_rep(self, state):
        # old version without projection code
        with tf.compat.v1.name_scope('graph_ops'):
            # Neighbors : [Num Relations, Nodes, Num Max Neighbors, Feat Dims]
            neighbors = tf.compat.v1.nn.embedding_lookup(state, self.rel_mat)
            mask_mat = tf.compat.v1.to_float(tf.compat.v1.expand_dims(tf.compat.v1.sequence_mask(self.rel_num, self.max_k), -1))

            if self.feat_attention:
                # exp_state [1, Nodes, 1, Feat dims]
                exp_state = tf.compat.v1.expand_dims(tf.compat.v1.expand_dims(state[1:], 1), 0)
                exp_state = tf.compat.v1.tile(exp_state, [self.num_relations, 1, self.max_k, 1])

                # Concatenated (Neightbors with state) :  [Num Relations, Nodes, Num Max Neighbors, 2*Feat Dims]
                att_x = tf.compat.v1.concat([neighbors, exp_state], -1)
                score = tf.compat.v1.layers.dense(inputs=att_x, units=1, name='state_attention')
                att_score = tf.compat.v1.nn.softmax(score, 2)
                rel_rep = tf.compat.v1.reduce_sum(neighbors*att_score, 2) / tf.compat.v1.expand_dims((tf.compat.v1.to_float(self.rel_num)+1e-10), -1)
            else:
                rel_rep = tf.compat.v1.reduce_sum(neighbors, 2) / tf.compat.v1.expand_dims((tf.compat.v1.to_float(self.rel_num)+1e-10), -1)
        return rel_rep

    def aggregate_relation_reps(self,):
        def to_input_shape(emb):
            # [R,N,K,D]
            emb_ = []
            for i in range(emb.shape[0]):
                exp = tf.compat.v1.tile(tf.compat.v1.expand_dims(emb[i], 0),[self.num_companies,1])
                emb_.append(tf.compat.v1.expand_dims(exp,0))
            return tf.compat.v1.concat(emb_,0)
        with tf.compat.v1.name_scope('aggregate_ops'):
            # all_rel_rep : [Num Relations, Nodes, Feat dims]
            if self.rel_attention:
                rel_emb = to_input_shape(self.rel_emb)
                att_x = tf.compat.v1.concat([self.all_rel_rep,rel_emb],-1)
                att_score = tf.compat.v1.nn.softmax(tf.compat.v1.layers.dense(inputs=att_x, units=1,
                                        name='relation_attention'), 1)
                updated_state = tf.compat.v1.reduce_mean(self.all_rel_rep * att_score, 0)
            else:
                updated_state = tf.compat.v1.reduce_mean(self.all_rel_rep, 0)
        return updated_state

    def create_relation_embedding(self, ):
        return tf.compat.v1.get_variable("Relation_embeddings", [self.num_relations, 32])

    def create_relation_onehot(self, ):
        one_hots = []
        for rel_idx in range(self.num_relations):
            one_hots.append(tf.compat.v1.one_hot([rel_idx],depth=self.num_relations))
        return tf.compat.v1.concat(one_hots,0)
    
    def to_input_shape(self, emb):
        emb_ = []
        for i in range(emb.shape[0]):
            exp = tf.compat.v1.tile(tf.compat.v1.expand_dims(tf.compat.v1.expand_dims(emb[i], 0),0),[self.num_companies, self.neighbors_sample,1])
            emb_.append(tf.compat.v1.expand_dims(exp,0))
        return tf.compat.v1.concat(emb_,0)

    def build_model(self):
        self.keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=())
        # x [num company, lookback]
        self.x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.config.lookback, self.input_dim])
        self.y = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.n_labels])
        self.rel_mat = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None, None, self.neighbors_sample]) # Edge, Node, Node
        self.rel_num = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None, None]) # Edge, Node
        self.max_k = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=())

        self.rel_emb = self.create_relation_onehot()

        self.exppanded = self.to_input_shape(self.rel_emb)

        state = self.get_state('lstm')
        # Graph operation
        self.all_rel_rep = self.get_relation_rep(state)

        # [Node, Feat dims]
        rel_summary = self.aggregate_relation_reps()
        updated_state = rel_summary+state[1:]

        logits = tf.compat.v1.layers.dense(inputs=updated_state, units=self.n_labels,
                                activation=tf.compat.v1.nn.leaky_relu, name='prediction')

        self.prob = tf.compat.v1.nn.softmax(logits)
        self.prediction = tf.compat.v1.argmax(logits, -1)

        with tf.compat.v1.name_scope("loss"):
            self.cross_entropy = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
            reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            loss = self.cross_entropy
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.compat.v1.control_dependencies(update_ops):
                self.train_step = tf.compat.v1.train.AdamOptimizer(self.config.lr).minimize(loss,
                                                                      global_step=self.global_step_tensor)

            correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(logits, -1), tf.compat.v1.argmax(self.y, -1))
            self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)
