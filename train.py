"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import math
import sys
import argparse
import random

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--se-plus', action='store_true')
parser.add_argument('--se1', help='add se block at sub1', action='store_true')
parser.add_argument('--se2', help='add se block at sub2', action='store_true')
parser.add_argument('--suffix', help='suffix')
args = parser.parse_args()

if args.se_plus:
    from model_mobile_plus import Model
else:
    from model_mobile import Model

with open('config.json') as config_file:
    config = json.load(config_file)

# Seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
train_tfrecords = config['train_tfrecords']
val_tfrecords = config['val_tfrecords']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
momentum = config['momentum']
tr_batch_size = config['training_batch_size']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Setting up the data and the model
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(100, args.se1, args.se2)
# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = tf.add_n([model.mean_xent, tf.multiply(weight_decay, model.weight_decay_loss)])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
        total_loss,
        global_step=global_step)


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir'] + args.suffix
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=1)
tf.summary.scalar('train accuracy', model.accuracy)
#tf.summary.image('images nat train', model.x_input, max_outputs=12)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)


def evaluate(sess):
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr_nat = 0
  data_dict = {
      'input_pipeline/batch_size:0': eval_batch_size
  }
  sess.run([model.data['val_init'], model.data['init_batch_size']], data_dict)
  for ibatch in range(num_batches):
    dict_nat = {
                model.is_training: False
                }
    cur_corr_nat = sess.run(model.num_correct,feed_dict = dict_nat)
    total_corr_nat += cur_corr_nat
  acc_nat = total_corr_nat / num_eval_examples
  summary_eval = tf.Summary(value=[
        tf.Summary.Value(tag='test accuracy', simple_value= acc_nat)])
  return summary_eval

with tf.Session() as sess:
  data_dict = {
      'input_pipeline/train_file:0': train_tfrecords,
      'input_pipeline/val_file:0': val_tfrecords,
      'input_pipeline/batch_size:0': tr_batch_size
  }
  sess.run([model.data['init_data'], model.data['init_batch_size']], data_dict)
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0
  sess.run(model.data['train_init'])

  # Main training loop
  for ii in range(max_num_training_steps):
    nat_dict = {
                model.is_training: True
                }
    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * tr_batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Actual training step
    start = timer()
    _ = sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start

    if (ii == 0) or ((ii+1) % num_summary_steps == 0):
      summary_eval = evaluate(sess)
      summary_writer.add_summary(summary_eval, global_step.eval(sess))
      data_dict = {'input_pipeline/batch_size:0': tr_batch_size}
      sess.run([model.data['train_init'], model.data['init_batch_size']], data_dict)

    # Write a checkpoint
    if (ii+1) % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)
