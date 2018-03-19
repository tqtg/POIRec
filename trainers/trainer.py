from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import math
from utils import metrics


class Trainer(BaseTrain):
  def __init__(self, sess, model, data, config, logger):
    super(Trainer, self).__init__(sess, model, data, config, logger)

  def train_epoch(self):
    print("Epoch: {}".format(self.sess.run(self.model.cur_epoch_tensor)))
    self.data.shuffle()
    losses = []
    total_corrects = 0
    self.offset = 0
    loop = tqdm(range(self.config.num_train_iter_per_epoch))
    for _ in loop:
      loop.set_description('Training')
      loss, num_corrects, batch_size = self.train_step()
      total_corrects += num_corrects
      if not math.isnan(loss):
        losses.append(loss)
      self.offset += batch_size
      loop.set_postfix(loss=loss)
    loss = np.mean(losses)
    precision_at_k = total_corrects / self.offset
    print("loss={:.4f}, precision@{}={:.4f}".format(loss, self.config.K, precision_at_k))

  def train_step(self):
    batch_x, batch_lengths, batch_y, batch_y_seq, batch_users, batch_size = next(
      self.data.next_train_batch(self.offset))
    # print(batch_x, batch_lengths, batch_y, batch_users)
    feed_dict = {self.model.location_sequences: batch_x,
                 self.model.sequence_lengths: batch_lengths,
                 self.model.next_locations: batch_y,
                 self.model.users: batch_users,
                 self.model.is_training: True}
    if self.config.seq2seq:
      feed_dict[self.model.labels_sequences] = batch_y_seq
    _, step, loss, top_k = self.sess.run([self.model.train_step,
                                          self.model.global_train_step_tensor,
                                          self.model.loss,
                                          self.model.top_k],
                                         feed_dict=feed_dict)
    # print("iter={}, loss={:.4f}, acc={:.4f}".format(step * batch_size, loss, accuracy))
    num_corrects = metrics.count_corrects(batch_y, top_k)
    if step % self.config.summarize_steps == 0:
      summaries_dict = {
        'loss': loss,
        # 'precision@{}'.format(self.config.K): num_corrects / batch_size
      }
      self.logger.summarize(step, summaries_dict=summaries_dict)
    return loss, num_corrects, batch_size

  def test(self):
    total_loss = 0
    total_corrects = 0
    self.offset = 0
    loop = tqdm(range(self.config.num_test_iters))
    for _ in loop:
      loop.set_description('Testing')
      loss, num_corrects, batch_size = self.test_step()
      total_loss += (loss * batch_size)
      total_corrects += num_corrects
      self.offset += batch_size
      loop.set_postfix(loss=loss)
    loss = total_loss / self.offset
    precision_at_k = total_corrects / self.offset
    print("loss={:.4f}, precision@{}={:.4f}".format(loss, self.config.K, precision_at_k))
    if precision_at_k > self.best_precision:
      self.best_loss = loss
      self.best_precision = precision_at_k
      self.best_epoch = self.sess.run(self.model.cur_epoch_tensor)
      # self.model.save(self.sess)
    print("best_loss={:.4f}, best_precision@{}={:.4f}, best_epoch={}\n".format(self.best_loss, self.config.K, self.best_precision, self.best_epoch))

  def test_step(self):
    batch_x, batch_lengths, batch_y, batch_y_seq, batch_users, batch_size = next(self.data.next_test_batch(self.offset))
    feed_dict = {self.model.location_sequences: batch_x,
                 self.model.sequence_lengths: batch_lengths,
                 self.model.next_locations: batch_y,
                 self.model.users: batch_users,
                 self.model.is_training: False}
    if self.config.seq2seq:
      feed_dict[self.model.labels_sequences] = batch_y_seq
    step, loss, top_k = self.sess.run([self.model.increment_test_step_tensor,
                                                                self.model.loss,
                                                                self.model.top_k],
                                                               feed_dict=feed_dict)
    # print("iter={}, loss={:.4f}, acc={:.4f}".format(step * batch_size, loss, accuracy))
    num_corrects = metrics.count_corrects(batch_y, top_k)
    if step % int(self.config.summarize_steps) == 0:
      summaries_dict = {
        'loss': loss,
        # 'precision@{}'.format(self.config.K): num_corrects / batch_size
      }
      self.logger.summarize(step, summarizer="test", summaries_dict=summaries_dict)
    return loss, num_corrects, batch_size
