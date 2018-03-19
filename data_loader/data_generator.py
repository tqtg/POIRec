import numpy as np
import random
import os

class DataGenerator:
  def __init__(self, config):
    self.config = config
    # load data here
    self.train_data = []
    self.test_data = []
    with open(os.path.join('data/', config.data_set, 'train.txt'), 'r') as data_file:
      for line in data_file:
        self.train_data.append(line.strip().split())
    with open(os.path.join('data/', config.data_set, 'test.txt'), 'r') as data_file:
      for line in data_file:
        self.test_data.append(line.strip().split())
    self.config.num_train_iter_per_epoch = int(len(self.train_data) / self.config.batch_size)
    if self.config.num_train_iter_per_epoch * self.config.batch_size < len(self.train_data):
      self.config.num_train_iter_per_epoch += 1
    self.config.num_test_iters = int(len(self.test_data) / self.config.batch_size)
    if self.config.num_test_iters * self.config.batch_size < len(self.test_data):
      self.config.num_test_iters += 1

  def next_train_batch(self, offset):
    yield self.next_batch(self.train_data, offset)

  def next_test_batch(self, offset):
    yield self.next_batch(self.test_data, offset)

  def next_batch(self, data, offset):
    first_idx = offset
    last_idx = offset + self.config.batch_size
    if last_idx > len(data):
      last_idx = -1
    batch_data = data[first_idx:last_idx]
    batch_size = len(batch_data)
    batch_max_length = max(map(len, batch_data)) - 2
    batch_x = np.zeros(shape=[batch_size, batch_max_length])
    batch_y = np.zeros(batch_size)
    batch_y_seq = np.zeros_like(batch_x)
    batch_lengths = np.zeros(batch_size)
    batch_users = np.zeros(batch_size)
    for i, sequence in enumerate(batch_data):
      batch_users[i] = int(sequence[0])
      batch_y[i] = int(sequence[-1])
      sequence = sequence[1:-1]
      batch_lengths[i] = len(sequence)
      for j, location in enumerate(sequence):
        batch_x[i, j] = int(location)
        if j > 0:
          batch_y_seq[i, j - 1] = int(location)
      batch_y_seq[i, len(sequence) - 1] = batch_y[i]
    return batch_x, batch_lengths, batch_y, batch_y_seq, batch_users, batch_size

  def shuffle(self):
    random.shuffle(self.train_data)