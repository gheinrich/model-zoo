"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N, classification_accuracy
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

import digitsDataPluginBAbI

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

### get data ###
plugin_data = digitsDataPluginBAbI.utils.parse_folder_phase(FLAGS.data_dir, FLAGS.task_id, train=True)
plugin_data_stats = digitsDataPluginBAbI.utils.get_stats(plugin_data)
print(plugin_data_stats)
n_plugin_entries = len(plugin_data)
sentence_size = plugin_data_stats['sentence_size']
memory_size = plugin_data_stats['story_size']
vocab_size = 20
data = []
data = np.empty((n_plugin_entries, 2, memory_size, sentence_size))
print('data.shape',data.shape)
labels = np.empty((n_plugin_entries, 1, memory_size, sentence_size))
for i, entry in enumerate(plugin_data):
    feature, label = digitsDataPluginBAbI.utils.encode_sample(
        entry,
        plugin_data_stats['word_map'],
        plugin_data_stats['sentence_size'],
        plugin_data_stats['story_size'])
    data[i] = feature
    labels[i] = label
n_train = 900
train_data = data[:n_train]
train_labels = labels[:n_train]
val_data = data[n_train:]
val_labels = labels[n_train:]
###


print("Longest sentence length", sentence_size)
print("Memory size", memory_size)
print("vocab size", vocab_size)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

#optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer)

    summary_writer = tf.train.SummaryWriter('/tmp/tf', sess.graph)

    for t in range(1, FLAGS.epochs+1):
        #np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            sq = train_data[start:end]
            a = train_labels[start:end]
            summaries, cost_t = model.batch_fit(sq, None, a)
            total_cost += cost_t
        summary_writer.add_summary(summaries, t)

        average_cost = total_cost / len(batches)

        if t % FLAGS.evaluation_interval == 0:            

            val_acc = model.accuracy(val_data, val_labels)
            train_acc = model.accuracy(train_data, train_labels)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Average Cost:', average_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

