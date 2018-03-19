from base.base_model import BaseModel
from models.attention import attention
from models.lstm import BNLSTMCell
import tensorflow as tf


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

    # Embedding layer
    location_embeddings = tf.get_variable('loc_embedding_matrix', [self.config.num_loc + 1, self.config.num_hidden])
    embedded_locations = tf.nn.embedding_lookup(location_embeddings, self.location_sequences)

    user_embeddings = tf.get_variable('user_embedding_matrix', [self.config.num_user + 1, self.config.num_hidden])
    embedded_users = tf.nn.embedding_lookup(user_embeddings, self.users)
    embedded_users = tf.expand_dims(embedded_users, axis=1)

    inputs = tf.multiply(embedded_users, embedded_locations)

    # RNN
    if self.config.cell == "LSTM":
      cell = BNLSTMCell(self.config.num_hidden, self.is_training)
      # c, h
      c_init_state = tf.get_variable('c_init_state',
                                     [1, self.config.num_hidden],
                                     initializer=tf.random_normal_initializer(stddev=0.1))
      h_init_state = tf.get_variable('h_init_state',
                                     [1, self.config.num_hidden],
                                     initializer=tf.random_normal_initializer(stddev=0.1))
      init_state = (tf.tile(c_init_state, [batch_size, 1]), tf.tile(h_init_state, [batch_size, 1]))
      outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                          sequence_length=self.sequence_lengths,
                                          dtype=tf.float32,
                                          initial_state=init_state)
    elif self.config.cell == "GRU":
      cell = tf.nn.rnn_cell.GRUCell(self.config.num_hidden)
      init_state = tf.get_variable('init_state',
                                   [1, self.config.num_hidden],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
      init_state = tf.tile(init_state, [batch_size, 1])
      outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
                                               sequence_length=self.sequence_lengths,
                                               dtype=tf.float32,
                                               initial_state=init_state)
    else:
      raise NotImplementedError

    # Add dropout, as the model otherwise quickly overfits
    # outputs = tf.layers.dropout(outputs, self.config.dropout_rate, training=self.is_training)

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
      # self.correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.next_locations)
      # self.num_corrects = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int32))
      # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
      _, self.top_k = tf.nn.top_k(logits, k=self.config.K)


  def init_saver(self):
    # here you initialize the tensorflow saver that will be used in saving the checkpoints.
    self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
