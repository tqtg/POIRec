from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import math

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
    acc = total_corrects / self.offset
    print("loss={:.4f}, acc={:.4f}".format(loss, acc))

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
    _, step, loss, accuracy, num_corrects = self.sess.run([self.model.train_step,
                                                           self.model.global_train_step_tensor,
                                                           self.model.loss,
                                                           self.model.accuracy,
                                                           self.model.num_corrects],
                                                          feed_dict=feed_dict)
    # print("iter={}, loss={:.4f}, acc={:.4f}".format(step * batch_size, loss, accuracy))
    if step % self.config.summarize_steps == 0:
      summaries_dict = {
        'train_loss': loss,
        'train_acc': accuracy,
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
    acc = total_corrects / self.offset
    print("loss={:.4f}, acc={:.4f}".format(loss, acc))
    if acc > self.best_acc:
      self.best_loss = loss
      self.best_acc = acc
      self.best_epoch = self.sess.run(self.model.cur_epoch_tensor)
      # self.model.save(self.sess)
    print("best_loss={:.4f}, best_acc={:.4f}, best_epoch={}\n".format(self.best_loss, self.best_acc, self.best_epoch))

  def test_step(self):
    batch_x, batch_lengths, batch_y, batch_y_seq, batch_users, batch_size = next(self.data.next_test_batch(self.offset))
    feed_dict = {self.model.location_sequences: batch_x,
                 self.model.sequence_lengths: batch_lengths,
                 self.model.next_locations: batch_y,
                 self.model.users: batch_users,
                 self.model.is_training: False}
    if self.config.seq2seq:
      feed_dict[self.model.labels_sequences] = batch_y_seq
    step, loss, accuracy, num_corrects = self.sess.run([self.model.increment_test_step_tensor,
                                                        self.model.loss,
                                                        self.model.accuracy,
                                                        self.model.num_corrects],
                                                       feed_dict=feed_dict)
    # print("iter={}, loss={:.4f}, acc={:.4f}".format(step * batch_size, loss, accuracy))
    if step % int(self.config.summarize_steps) == 0:
      summaries_dict = {
        'test_loss': loss,
        'test_acc': accuracy,
      }
      self.logger.summarize(step, summarizer="test", summaries_dict=summaries_dict)
    return loss, num_corrects, batch_size
