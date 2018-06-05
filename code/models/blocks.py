######################################################################
# Basic Network Building Blocks
#
# Some of the functions are taken from the open source repositories
# of TensorFlow and OpenAI, as well as from Danijar Hafner blog.
#   - https://github.com/openai/baselines
#   - https://danijar.com/variable-sequence-lengths-in-tensorflow/
#
######################################################################


###################################
# Imports
#

import tensorflow as tf
import numpy as np


###################################
# Functions
#

def seqlen(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.
    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    with tf.name_scope('seq_len'):
        populated = tf.sign(tf.abs(sequence))
        length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
        mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)

    return length, mask


def pathlen(path):
    with tf.name_scope('path_len'):
        populated = tf.sign(tf.reduce_sum(tf.abs(path), 2))
        length = tf.cast(tf.reduce_sum(populated, 1), tf.int32)
        mask = tf.cast(populated, tf.float32)

    return length, mask


def get_padded(seq, seq_len, val):
    return np.pad(seq, (0, seq_len-len(seq)), 'constant', constant_values=(val,))


def last_relevant(output, length):
    with tf.name_scope('last_relevant'):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, 1])
        relevant = tf.squeeze(tf.gather(flat, index))

    return relevant


def mask_loss(total_loss, mask):
    with tf.name_scope('mask_loss'):
        loss = tf.reduce_sum(total_loss * mask, axis=1)

    return loss


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    with tf.variable_scope("minimize_and_clip"):
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        grad_norm = tf.global_norm([g[0] for g in gradients])
        return optimizer.apply_gradients(gradients), grad_norm


def birnn(inputs, dim, keep_prob, seq_len, name):
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            cell_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
            # cell_fwd = tf.nn.rnn_cell.DropoutWrapper(cell_fwd, output_keep_prob=keep_prob)
        with tf.variable_scope('backward' + name):
            cell_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)
            # cell_bwd = tf.nn.rnn_cell.DropoutWrapper(cell_bwd, output_keep_prob=keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fwd, cell_bw=cell_bwd, inputs=inputs,
                                                          sequence_length=seq_len, dtype=tf.float32, scope=name)
    return outputs, states


def rnn(inputs, dim, keep_prob, seq_lens, name):
    with tf.name_scope(name):
        cell_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        # cell_fwd = tf.nn.rnn_cell.DropoutWrapper(cell_fwd, output_keep_prob=keep_prob)
        outputs, state = tf.nn.dynamic_rnn(cell=cell_fwd, inputs=inputs,
                                           sequence_length=seq_lens, dtype=tf.float32, scope=name)

    return outputs, state


# based on code from:
# https://github.com/google/seq2seq/blob/master/seq2seq/decoders/attention.py
def attention_layer(query, keys, values, values_length, num_units, reuse=False):
    """ Computes attention scores and outputs.
    Args:
      query: The query used to calculate attention scores.
        In seq2seq this is typically the current state of the decoder.
        A tensor of shape `[B, ...]`
      keys: The keys used to calculate attention scores. In seq2seq, these
        are typically the outputs of the encoder and equivalent to `values`.
        A tensor of shape `[B, T, ...]` where each element in the `T`
        dimension corresponds to the key for that value.
      values: The elements to compute attention over. In seq2seq, this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`.
      values_length: An int32 tensor of shape `[B]` defining the sequence
        length of the attention values.
    Returns:
      A tuple `(scores, context)`.
      `scores` is vector of length `T` where each element is the
      normalized "score" of the corresponding `inputs` element.
      `context` is the final attention layer output corresponding to
      the weighted inputs.
      A tensor fo shape `[B, input_dim]`.
    """
    with tf.variable_scope('attention_layer', reuse=reuse):
        values_depth = values.get_shape().as_list()[-1]

        # Fully connected layers to transform both keys and query
        # into a tensor with 'num_units' units
        att_keys = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=num_units,
            activation_fn=None,
            scope="att_keys")
        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=num_units,
            activation_fn=None,
            scope="att_query")

        scores = tf.reduce_sum(att_keys * tf.expand_dims(att_query, 1), [2])

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, values_depth])

        return scores_normalized, context


def dense(x, size, name, activation=None, bias=True, weight_init=None, bias_init=None):
    w = tf.get_variable(name + "/w", [x.get_shape()[-1], size], initializer=weight_init)
    ret = tf.matmul(x, w)

    if bias:
        bias_init = tf.zeros_initializer() if bias_init is None else bias_init
        b = tf.get_variable(name + "/b", [size], initializer=bias_init)
        ret = ret + b

    if activation is not None:
        return activation(ret)
    else:
        return ret


def ffnn2l(inputs, scope, size=128, activation=tf.nn.relu, weight_init=None, bias_init=tf.zeros_initializer()):
    with tf.variable_scope(scope):
        layer1 = tf.layers.dense(inputs=inputs, units=size, activation=activation, kernel_initializer=weight_init, bias_initializer=bias_init)
        layer2 = tf.layers.dense(inputs=layer1, units=size, activation=activation, kernel_initializer=weight_init, bias_initializer=bias_init)
        return layer2


def ffnn(inputs, scope, n_layers=2, size=128, activation=tf.nn.relu):
    with tf.variable_scope(scope):
        layers = [None] * n_layers
        layers[0] = tf.layers.dense(inputs=inputs, units=size, activation=activation)
        for i in range(1, n_layers):
            layers[i] = tf.layers.dense(inputs=layers[i-1], units=size, activation=activation)

        return layers[-1]


def conv_max_pool(inputs, num_filters, filter_size, scope_name):
    with tf.variable_scope("conv-maxpool-{}-{}".format(filter_size, scope_name)):
        num_channels = inputs.get_shape()[3]
        filter_shape = [1, filter_size, num_channels, num_filters]
        filter_ = tf.get_variable("filter", shape=filter_shape, dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[num_filters], dtype=tf.float32)
        strides = [1, 1, 1, 1]

        conv = tf.nn.conv2d(inputs, filter_, strides=strides, padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, bias))

        rank = len(h.shape) - 2
        pooled = tf.reduce_max(h, axis=rank)

        return pooled
