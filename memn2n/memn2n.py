"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def classification_loss(pred, y):
    """
    Definition of the loss for regular classification
    """
    ssoftmax = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y, name='cross_entropy_single')
    return tf.reduce_mean(ssoftmax, name='cross_entropy_batch')

def classification_accuracy(pred, y):
    """
    Default definition of accuracy. Something is considered accurate if and only
    if a true label exactly matches the highest value in the prediction vector.
    """
    single_acc = tf.equal(y, tf.argmax(pred, 1))
    return tf.reduce_mean(tf.cast(single_acc, tf.float32), name='accuracy')

def dig_memn2n(x, embeddings, weights, encoding, hops):
    """
    Create model
    """

    with tf.variable_scope("memn2n"):
        # x has shape [batch_size, story_size, sentence_size, 2]
        # unpack along last dimension to extract stories and questions
        x = tf.transpose(x, [0, 2, 3, 1])

        stories, questions = tf.unpack(x, axis=3)

        questions = tf.Print(questions, [questions], message="q: ", summarize=20, first_n=10)

        # assume single sentence in question
        questions = questions[:, 0, :]

        q_emb = tf.nn.embedding_lookup(embeddings['B'], questions, name='q_emb')
        u_0 = tf.reduce_sum(q_emb * encoding, 1)
        u = [u_0]
        for _ in xrange(hops):
            m_emb = tf.nn.embedding_lookup(embeddings['A'], stories, name='m_emb')
            m = tf.reduce_sum(m_emb * encoding, 2) + weights['TA']
            # hack to get around no reduce_dot
            u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
            dotted = tf.reduce_sum(m * u_temp, 2)

            # Calculate probabilities
            probs = tf.nn.softmax(dotted)

            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            c_temp = tf.transpose(m, [0, 2, 1])
            o_k = tf.reduce_sum(c_temp * probs_temp, 2)

            u_k = tf.matmul(u[-1], weights['H']) + o_k

            u.append(u_k)

        o = tf.matmul(u_k, weights['W'])
    return o


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        encoding=position_encoding,
        session=tf.Session(),
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name

        self._build_inputs()
        self._build_vars()
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        ###
        dict_size = vocab_size
        story_size = memory_size
        with tf.variable_scope(self._name):

            embeddings = {
                'A': tf.get_variable('A', [dict_size, embedding_size], initializer=initializer),
                'B': tf.get_variable('B', [dict_size, embedding_size], initializer=initializer),
            }
            weights = {
                'TA': tf.get_variable('TA', [story_size, embedding_size], initializer=initializer),
                'H': tf.get_variable('H', [embedding_size, embedding_size], initializer=initializer),
                'W': tf.get_variable('W', [embedding_size, dict_size], initializer=initializer),
            }
            x = self._stories_queries
            net_output = dig_memn2n(x, embeddings, weights, self._encoding, hops)

            tf.histogram_summary("A_hist", embeddings['A'])
            tf.histogram_summary("X_hist", x)

        model = net_output
        ###

        def loss(y):
            y = tf.to_int64(y[:, 0, 0, 0], name='y_int')
            tf.histogram_summary("Y_hist", y)
            loss = classification_loss(model, y)
            return loss

        # loss op
        loss_op = loss(self._answers)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        #grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        #grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")


        def accuracy(y):
            y = tf.to_int64(y[:, 0, 0, 0], name='y_accuracy')
            acc = classification_accuracy(model, y)
            return acc

        accuracy_op = accuracy(self._answers)

        # assign ops
        self.loss_op = loss_op
        self.accuracy_op = accuracy_op
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

        self._merged_summaries = tf.merge_all_summaries()


    def _build_inputs(self):
        self._stories_queries = tf.placeholder(tf.int32, [None, 2, self._memory_size, self._sentence_size], name="stories")
        self._answers = tf.placeholder(tf.int32, [None, 1, self._memory_size, self._sentence_size], name="answers")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            pass
        self._nil_vars = set([]) # set([self.A.name, self.B.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            for _ in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb * self._encoding, 2) + self.TA
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # nonlinearity
                if self._nonlin:
                    u_k = nonlin(u_k)

                u.append(u_k)

            return tf.matmul(u_k, self.W)

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories_queries: stories, self._answers: answers}
        summaries, loss, _ = self._sess.run([self._merged_summaries, self.loss_op, self.train_op], feed_dict=feed_dict)
        return summaries, loss

    def accuracy(self, stories_queries, answers):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_queries: stories_queries, self._answers: answers}
        return self._sess.run(self.accuracy_op, feed_dict=feed_dict)


