import os
from math import pi

import numpy as np
import tensorflow as tf

from base.base_model import BaseModel
from models.attention import attention
from models.lstm import BNLSTMCell


class Model(BaseModel):
  def __init__(self, config):
    super(Model, self).__init__(config)

    self.build_model()
    self.init_saver()

  def build_model(self):
    # here you build the tensorflow graph of any model you want and also define the loss.
    self.is_training = tf.placeholder(tf.bool)

    self.location_sequences = tf.placeholder(tf.int32, shape=[None, None])
    self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
    self.next_locations = tf.placeholder(tf.int32, shape=[None])
    self.users = tf.placeholder(tf.int32, shape=[None])

    batch_size = tf.shape(self.sequence_lengths)[0]

    location_embeddings = tf.get_variable('loc_embeddings', [self.config.num_loc + 1, self.config.num_hidden])
    embedded_locations = tf.nn.embedding_lookup(location_embeddings, self.location_sequences)

    user_embeddings = tf.get_variable('user_embeddings', [self.config.num_user + 1, self.config.num_hidden])
    embedded_users = tf.nn.embedding_lookup(user_embeddings, self.users)
    embedded_users = tf.expand_dims(embedded_users, axis=1)
    inputs = tf.multiply(embedded_users, embedded_locations)

    # RNN
    if self.config.cell == "LSTM":
      cell = BNLSTMCell(self.config.num_hidden, self.is_training)
      # c, h
      c_init_state = tf.get_variable('c_init_state',
                                     [1, self.config.num_hidden],
                                     initializer=tf.constant_initializer(0.0))
      h_init_state = tf.get_variable('h_init_state',
                                     [1, self.config.num_hidden],
                                     initializer=tf.constant_initializer(0.0))
      init_state = (tf.tile(c_init_state, [batch_size, 1]), tf.tile(h_init_state, [batch_size, 1]))
      outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                          sequence_length=self.sequence_lengths,
                                          dtype=tf.float32,
                                          initial_state=init_state)
    elif self.config.cell == "GRU":
      cell = tf.nn.rnn_cell.GRUCell(self.config.num_hidden)
      init_state = tf.get_variable('init_state',
                                   [1, self.config.num_hidden],
                                   initializer=tf.constant_initializer(0.0))
      init_state = tf.tile(init_state, [batch_size, 1])
      outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
                                               sequence_length=self.sequence_lengths,
                                               dtype=tf.float32,
                                               initial_state=init_state)
    else:
      raise NotImplementedError

    # Add dropout, as the model otherwise quickly overfits
    outputs = tf.layers.dropout(outputs, self.config.dropout_rate, training=self.is_training)

    # next location indices
    idx = tf.range(batch_size) * tf.shape(outputs)[1] + (self.sequence_lengths - 1)

    if self.config.seq2seq:
      self.labels_sequences = tf.placeholder(tf.int32, shape=[None, None])
      labels = tf.reshape(self.labels_sequences, [-1])
      outputs = tf.reshape(outputs, [-1, self.config.num_hidden])
      logits = tf.matmul(outputs, location_embeddings, transpose_b=True)
    else:
      if self.config.attention:
        final_output = attention(outputs, self.config.num_hidden)
      else:
        final_output = tf.gather(tf.reshape(outputs, [-1, self.config.num_hidden]), idx)
      logits = tf.matmul(final_output, location_embeddings, transpose_b=True)
      labels = self.next_locations

    with tf.name_scope("loss"):
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      if self.config.seq2seq:
        seqlen_mask = tf.sequence_mask(self.sequence_lengths, dtype=tf.float32)
        cross_entropy = cross_entropy * tf.reshape(seqlen_mask, [-1])
        self.loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(seqlen_mask)
      else:
        self.loss = tf.reduce_mean(cross_entropy)

      # Optimization
      optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
      # Gradient clipping
      gvs = optimizer.compute_gradients(self.loss)
      capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      # Update parameters
      self.train_step = optimizer.apply_gradients(capped_gvs, global_step=self.global_train_step_tensor)

    with tf.name_scope("acc"):
      # count correct only for the last location in the sequences
      if self.config.seq2seq:
        logits = tf.gather(logits, idx)
      predictions = tf.nn.softmax(logits)

      # get predictions within distance_limit
      geo = tf.get_variable('geo', shape=[self.config.num_loc + 1, 2],
                            initializer=tf.constant_initializer(self.read_geo()),
                            trainable=False) # [n, 2]
      location_geo = tf.nn.embedding_lookup(geo, self.location_sequences) # [b, l, 2]
      mean_geo = tf.divide(tf.reduce_sum(location_geo, axis=1, keep_dims=True),
                           tf.reshape(tf.cast(self.sequence_lengths, tf.float32), [batch_size, 1, 1])) # [b, 1, 2]
      geo = tf.expand_dims(geo, axis=0) # [1, n, 2]

      distances = self.haversine(geo, mean_geo)
      mask = tf.less_equal(distances, tf.constant(self.config.distance_limit, dtype=tf.float32))

      predictions = tf.multiply(predictions, tf.cast(mask, tf.float32))
      _, self.top_k = tf.nn.top_k(predictions, k=self.config.K)

      # self.correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.next_locations)
      # self.num_corrects = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int32))
      # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


  def haversine(self, mean_geo, geo):
    lat1, lon1 = tf.split(geo, num_or_size_splits=2, axis=2)  # [1, n, 1]
    lat2, lon2 = tf.split(mean_geo, num_or_size_splits=2, axis=2)  # [b, 1, 1]

    dlat = self.radians(tf.subtract(lat2, lat1))  # [b, n, 1]
    dlon = self.radians(tf.subtract(lon2, lon1))  # [b, n, 1]

    a = tf.sin(dlat / 2) * tf.sin(dlat / 2) + tf.cos(self.radians(lat1)) \
        * tf.cos(self.radians(lat2)) * tf.sin(dlon / 2) * tf.sin(dlon / 2)
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))

    radius = tf.constant(6371, dtype=tf.float32)
    d = radius * c  # [b, n, 1]
    return tf.reshape(d, [-1, self.config.num_loc + 1])  # [b, n]


  def radians(self, value):
    return tf.multiply(value, tf.constant(pi / 180))


  def read_geo(self):
    print('Reading geo data...\n')
    geo_matrix = np.zeros([self.config.num_loc + 1, 2])
    with open(os.path.join('data/', self.config.data_set, "geo.txt")) as f:
      for line in f:
        tokens = line.strip().split()
        geo_matrix[int(tokens[0]), 0] = float(tokens[1])
        geo_matrix[int(tokens[0]), 1] = float(tokens[2])
    return geo_matrix


  def init_saver(self):
    # here you initialize the tensorflow saver that will be used in saving the checkpoints.
    self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
