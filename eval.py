from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import sys

import numpy as np
import tensorflow as tf

import tqdm
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
model = Model(100, args.se1, args,se2)

saver = tf.train.Saver(max_to_keep=5)

def evaluate(sess):
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr_nat = 0
  sess.run(model.data['val_init'])
  for ibatch in range(num_batches):
    dict_nat = {
                model.is_training: False
                }
    cur_corr_nat = sess.run(model.num_correct,feed_dict = dict_nat)
    total_corr_nat += cur_corr_nat
  acc_nat = total_corr_nat / num_eval_examples
  print('model dir: '+args.model_dir)
  print(f'acc_nat: {acc_nat}')


with tf.Session() as sess:
    data_dict = {
        'input_pipeline/train_file:0': train_tfrecords,
        'input_pipeline/val_file:0': val_tfrecords,
        'input_pipeline/batch_size:0': eval_batch_size
    }
    sess.run([model.data['init_data'], model.data['init_batch_size']], data_dict)
    sess.run(tf.global_variables_initializer())
    # restore
    cur_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    saver.restore(sess, cur_checkpoint)
    # evaluate
    evaluate(sess)
