######################################################################
# Neural Model
#
# Dueling Deep Q-Network
#
######################################################################


###################################
# Imports
#

import tensorflow as tf
import numpy as np
from models.replay_buffer import State, StateExt
from models.blocks import *


###################################
# Classes
#

class RLModel(object):
    def __init__(self, known_emb, unknown_emb, char_emb_len, model_conf, scope):
        # hyperparameters
        self.word_embedding_dim = model_conf.word_embedding_dim
        self.char_embedding_dim = model_conf.char_embedding_dim
        self.hidden_dim_q = model_conf.hidden_dim_q
        self.hidden_dim_x = model_conf.hidden_dim_x
        self.hidden_dim_a = model_conf.hidden_dim_a
        self.props_dim = model_conf.props_dim
        self.ans_props_dim = model_conf.ans_props_dim
        self.output_dim = model_conf.output_dim
        self.token_length = model_conf.token_length
        self.observ_length = model_conf.observ_length
        self.learning_rate = model_conf.learning_rate
        self.dropout_rate = model_conf.dropout_rate

        self.kernel_initializer = tf.glorot_uniform_initializer()
        self.bias_initializer = tf.truncated_normal_initializer(mean=0.011, stddev=0.005)

        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model(known_emb, unknown_emb, char_emb_len)

    def _build_model(self, known_emb, unknown_emb, char_emb_len):
        # placeholders
        self.q_w = tf.placeholder(tf.int32, [None, None], name='q_w')
        self.q_c = tf.placeholder(tf.int32, [None, None, self.token_length], name='q_c')
        self.x_w = tf.placeholder(tf.int32, [None, self.observ_length], name='x_w')
        self.x_c = tf.placeholder(tf.int32, [None, self.observ_length, self.token_length], name='x_c')
        self.p = tf.placeholder(tf.int32, [None, self.props_dim], name='p')
        self.a_w = tf.placeholder(tf.int32, [None, None], name='a_w')
        self.a_c = tf.placeholder(tf.int32, [None, None, self.token_length], name='a_c')
        self.a_p = tf.placeholder(tf.float32, [None, self.ans_props_dim], name='a_p')

        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.y = tf.placeholder(tf.float32, [None], name='y')
        self.w = tf.placeholder(tf.float32, [None], name='w')
        self.do = tf.placeholder_with_default(1.0, shape=())

        # get lengths of unpadded sentences
        q_seq_length, q_seq_mask = seqlen(self.q_w)
        x_seq_lengths, x_seq_mask = seqlen(self.x_w)
        a_seq_lengths, a_seq_mask = seqlen(self.a_w)

        # word embedding lookup and dropout at embedding layer
        self.EWk = tf.Variable(known_emb, name='EWk', trainable=False)
        self.EWu = tf.Variable(unknown_emb, name='EWu', trainable=True)
        self.EW = tf.concat([self.EWk, self.EWu], axis=0)

        qw_emb = tf.nn.embedding_lookup(self.EW, self.q_w)
        qw_emb_drop = tf.nn.dropout(qw_emb, self.do)
        xw_emb = tf.nn.embedding_lookup(self.EW, self.x_w)
        xw_emb_drop = tf.nn.dropout(xw_emb, self.do)
        aw_emb = tf.nn.embedding_lookup(self.EW, self.a_w)
        aw_emb_drop = tf.nn.dropout(aw_emb, self.do)

        # character embedding lookup and dropout at embedding layer
        # +1 for uncommon characters
        self.ECku = tf.get_variable("ECku", (char_emb_len+1, self.char_embedding_dim), tf.float32, initializer=self.kernel_initializer)
        self.EC = tf.concat([tf.zeros((1, self.char_embedding_dim), dtype=tf.float32),
                             self.ECku], axis=0, name='EC')

        qc_emb = tf.nn.embedding_lookup(self.EC, self.q_c)
        qc_emb_drop = tf.nn.dropout(qc_emb, self.do)
        qc_pooled = conv_max_pool(qc_emb_drop, 100, 5, "qc_pooled")
        xc_emb = tf.nn.embedding_lookup(self.EC, self.x_c)
        xc_emb_drop = tf.nn.dropout(xc_emb, self.do)
        xc_pooled = conv_max_pool(xc_emb_drop, 100, 5, "xc_pooled")
        ac_emb = tf.nn.embedding_lookup(self.EC, self.a_c)
        ac_emb_drop = tf.nn.dropout(ac_emb, self.do)
        ac_pooled = conv_max_pool(ac_emb_drop, 100, 5, "ac_pooled")

        # BiLSTM layer - q
        qw_qc_concat = tf.concat([qw_emb_drop, qc_pooled], 2)
        q_outputs, q_states = birnn(qw_qc_concat, dim=self.hidden_dim_q, keep_prob=self.do,
                                    seq_len=q_seq_length, name='q_bilstm')
        # q_states_concat = tf.concat([q_states[0][1], q_states[1][1]], 1, name='q_states_concat')
        q_outputs_concat = tf.concat(q_outputs, 2, name='q_outputs_concat')
        self.h2q_outputs = ffnn2l(q_outputs_concat, "h2q_outputs", size=256, activation=tf.nn.relu)
        q_weights = tf.layers.dense(inputs=self.h2q_outputs, units=1, activation=None, name='q_weights')
        self.q_weights_norm = tf.nn.softmax(tf.squeeze(q_weights, [2]), name="q_weights_norm")
        self.q_weighted = tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_weights_norm, -1), q_outputs_concat), axis=1, name="q_weighted")

        # LSTM layer - x
        xw_xc_concat = tf.concat([xw_emb_drop, xc_pooled], 2)
        x_outputs, x_state = rnn(xw_xc_concat, dim=self.hidden_dim_x, keep_prob=self.do,
                                 seq_lens=x_seq_lengths, name='x_lstm')
        self.h2x_outputs = ffnn2l(x_outputs, "h2x_outputs", size=256, activation=tf.nn.relu)
        x_weights = tf.layers.dense(inputs=self.h2x_outputs, units=1, activation=None, name='x_weights')
        self.x_weights_norm = tf.nn.softmax(tf.squeeze(x_weights, [2]), name="x_weights_norm")
        self.x_weighted = tf.reduce_sum(tf.multiply(tf.expand_dims(self.x_weights_norm, -1), x_outputs), axis=1, name="x_weighted")

        # LSTM layer - a
        aw_ac_concat = tf.concat([aw_emb_drop, ac_pooled], axis=2)
        a_outputs, a_state = rnn(aw_ac_concat, dim=self.hidden_dim_a, keep_prob=self.do,
                                 seq_lens=a_seq_lengths, name='a_lstm')
        # c:= a_state[0], h:= a_state[1]
        a_repr = tf.concat([a_state[1], self.a_p], axis=1)

        # concatenate encoded observations
        self.h0 = tf.concat([self.q_weighted, self.x_weighted, a_repr], axis=1, name='h0')
        self.h1 = tf.layers.dense(inputs=self.h0, units=512, activation=tf.nn.relu, name='h1',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)

        # dueling
        self.h2v = tf.layers.dense(inputs=self.h1, units=256, activation=tf.nn.relu, name='h2v',
                                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        self.h2a = tf.layers.dense(inputs=self.h1, units=256, activation=tf.nn.relu, name='h2a',
                                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)

        # concatenate encoded observations with node props
        props = tf.cast(self.p, tf.float32)
        self.h2v_props = tf.concat([self.h2v, props], 1, name='h2v_props')
        self.h2a_props = tf.concat([self.h2a, props], 1, name='h2a_props')

        self.values = tf.layers.dense(inputs=self.h2v_props, units=1, activation=None, name='values')
        self.advantages = tf.layers.dense(inputs=self.h2a_props, units=self.output_dim, activation=None, name='advantages')

        self.preds = self.values + (self.advantages -
                                    tf.reduce_mean(self.advantages, reduction_indices=1, keep_dims=True))

        # loss calculation
        flat_preds = tf.reshape(self.preds, [-1])
        batch_size = tf.shape(self.x_w)[0]
        gather_indices = tf.range(batch_size) * self.output_dim + self.a
        partitions = tf.reduce_sum(tf.one_hot(gather_indices, tf.shape(flat_preds)[0], dtype='int32'), 0)
        self.a_preds = tf.dynamic_partition(flat_preds, partitions, 2)[1]

        self.td_err = self.a_preds - self.y
        losses = huber_loss(self.td_err)
        self.weighted_loss = tf.reduce_mean(self.w * losses)

        # GD with Adam
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, momentum=0.0, epsilon=1e-06)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.train_op, self.grad_norm = minimize_and_clip(self.optimizer, self.weighted_loss, scope_vars)

        # visualization
        self.summaries = self.get_summaries()

    def predict(self, sess, states, get_weights=False):
        feed = self.get_feed_dict(states)
        if get_weights:
            return sess.run([self.preds, self.x_weights_norm, self.q_weights_norm], feed)
        else:
            return sess.run(self.preds, feed)

    def update(self, sess, states, actions, targets, weights, writer, step):
        feed = self.get_feed_dict_batch(states, actions, targets, weights)

        if writer is None:
            _, grads, loss, td_err = sess.run([self.train_op, self.grad_norm, self.weighted_loss, self.td_err], feed)
        else:
            _, grads, loss, td_err, summaries = sess.run([self.train_op, self.grad_norm, self.weighted_loss,
                                                          self.td_err, self.summaries], feed)
            writer.add_summary(summaries, step)

        return grads, loss, td_err

    def get_feed_dict(self, states):
        if isinstance(states, State) or isinstance(states, StateExt):
            feed_dict = {self.q_w: np.vstack([states.q_w]),
                         self.q_c: states.q_c,
                         self.x_w: states.x_w,
                         self.x_c: states.x_c,
                         self.p: states.p,
                         self.a_w: np.zeros(shape=(1, 1), dtype=np.float32),
                         self.a_c: np.zeros(shape=(1, 1, self.token_length), dtype=np.float32),
                         self.a_p: np.zeros(shape=(1, self.ans_props_dim), dtype=np.float32),
                         self.do: self.dropout_rate}
            if isinstance(states, StateExt):
                feed_dict.update({self.a_w: np.vstack([states.a_w]),
                                  self.a_c: states.a_c,
                                  self.a_p: states.a_p})

        else:
            qs_w = [state.q_w for state in states]
            max_q_len = max([len(q) for q in qs_w])
            qs_w = [get_padded(q, max_q_len, 0) for q in qs_w]
            qs_c = [np.reshape(np.concatenate(
                    [state.q_c[0], np.zeros((max_q_len-len(state.q_c[0]), self.token_length), dtype=np.int32)]),
                    (1, max_q_len, self.token_length))
                    for state in states]

            as_w = [state.a_w if isinstance(state, StateExt) else np.zeros(shape=1, dtype=np.float32)
                    for state in states]
            as_c = [state.a_c if isinstance(state, StateExt) else np.zeros(shape=(1, 1, self.token_length), dtype=np.float32)
                    for state in states]
            as_p = [state.a_p[0] if isinstance(state, StateExt) else np.zeros(shape=self.ans_props_dim, dtype=np.float32)
                    for state in states]
            max_a_len = max([len(a) for a in as_w])
            as_w = [get_padded(a, max_a_len, 0) for a in as_w]
            as_c = [np.reshape(np.concatenate(
                    [a[0], np.zeros((max_a_len - len(a[0]), self.token_length), dtype=np.int32)]),
                    (1, max_a_len, self.token_length))
                    for a in as_c]

            feed_dict = {self.q_w: np.vstack(qs_w),
                         self.q_c: np.vstack(qs_c),
                         self.x_w: np.vstack([state.x_w[0] for state in states]),
                         self.x_c: np.vstack([state.x_c for state in states]),
                         self.p: np.vstack([state.p[0] for state in states]),
                         self.a_w: np.vstack(as_w),
                         self.a_c: np.vstack(as_c),
                         self.a_p: np.vstack(as_p),
                         self.do: self.dropout_rate}

        return feed_dict

    def get_feed_dict_batch(self, states_batch, actions_batch, targets_batch, weights):
        feed_dict = self.get_feed_dict(states_batch)
        feed_dict.update({self.a: np.array(actions_batch),
                          self.y: np.array(targets_batch),
                          self.w: weights})

        return feed_dict

    def get_num_model_params(self):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        total_params = 0
        for variable in scope_vars:
            shape = variable.get_shape()
            variable_params = 1
            for dim in shape:
                variable_params *= dim.value
            total_params += variable_params

        return total_params

    def get_summaries(self):
        scope_hists = self.get_var_weights()
        summaries = tf.summary.merge(scope_hists + [
            tf.summary.histogram("h1_act", self.h1),
            tf.summary.histogram("h2v_act", self.h2v),
            tf.summary.histogram("h2a_act", self.h2a),
            tf.summary.histogram("values_act", self.values),
            tf.summary.histogram("advantages_act", self.advantages),
            tf.summary.histogram('preds', self.preds),
            tf.summary.scalar("avg.loss", self.weighted_loss)
        ])

        return summaries

    def get_var_weights(self):
        scope_hists = []
        scope_names = ['h1', 'h2v', 'h2a', 'values', 'advantages']
        for scope_name in scope_names:
            scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/' + scope_name)
            scope_hists.extend([tf.summary.histogram(scope_name + '_kernel', scope_vars[0]),
                                tf.summary.histogram(scope_name + '_bias', scope_vars[1])])

        return scope_hists

    def write_weights_log(self, sess, step, writer):
        summaries = tf.summary.merge(self.get_var_weights())
        writer.add_summary(sess.run(summaries, {}), step)


class ModelParametersCopier:
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)
