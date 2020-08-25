from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import sys

import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--se-plus', action='store_true')
parser.add_argument('--se1', help='add se block at sub1', action='store_true')
parser.add_argument('--se2', help='add se block at sub2', action='store_true')
parser.add_argument('--model-dir', type=str)
args = parser.parse_args()

if args.se_plus:
    from model_mobile_plus import Model
else:
    from model_mobile import Model

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
train_tfrecords = config['train_tfrecords']
val_tfrecords = config['val_tfrecords']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Setting up the data and the model
model = Model(100, args.se1, args.se2)

saver = tf.train.Saver(max_to_keep=5)
tf.summary.image('images nat train', model.x_input, max_outputs=12)
merged_summaries = tf.summary.merge_all()

def evaluate(sess):
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr_nat = 0
  for ibatch in range(num_batches):
    dict_nat = {
                model.is_training: False
                }
    summary, cur_corr_nat = sess.run([merged_summaries, model.num_correct],feed_dict = dict_nat)
    total_corr_nat += cur_corr_nat
    summary_writer.add_summary(summary)
  acc_nat = total_corr_nat / num_eval_examples
  print(f'acc_nat: {acc_nat}')


with tf.Session() as sess:
    data_dict = {
        'input_pipeline/train_file:0': train_tfrecords,
        'input_pipeline/val_file:0': val_tfrecords,
        'input_pipeline/batch_size:0': eval_batch_size
    }
    sess.run([model.data['init_data'], model.data['init_batch_size']], data_dict)
    summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(model.data['val_init'])
    # restore
    cur_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    print(cur_checkpoint)
    saver.restore(sess, cur_checkpoint)
    # evaluate
    evaluate(sess)
