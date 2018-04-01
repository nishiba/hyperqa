# coding: utf-8

import tensorflow as tf
import numpy as np


class HyperQA(object):
    def __init__(self,
                 sequence_length: int,
                 projection_dim: int,
                 margin: float,
                 embedding: np.ndarray) -> None:
        # parameters
        self.sequence_length = sequence_length
        self.projection_dim = projection_dim
        self.margin = margin

        # placeholder
        self.input_positive_answer = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.input_negative_answer = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.input_question = tf.placeholder(tf.int32, [None, self.sequence_length])

        # layers
        self.embedding = tf.keras.layers.Embedding(
            embedding.shape[0], embedding.shape[1], trainable=False, embeddings_initializer=tf.constant_initializer(embedding))
        self.projection = tf.keras.layers.Dense(self.projection_dim)
        self.scoring = tf.keras.layers.Dense(1)

        # representations
        self.positive_answer_representation = self._to_representation(self.input_positive_answer)
        self.negative_answer_representation = self._to_representation(self.input_negative_answer)
        self.question_representation = self._to_representation(self.input_question)

        # hyperbolic distance
        self.p_distance = self._hyperbolic_distance(self.positive_answer_representation, self.question_representation)
        self.n_distance = self._hyperbolic_distance(self.negative_answer_representation, self.question_representation)

        # score
        self.p_score = self.scoring(self.p_distance)
        self.n_score = self.scoring(self.n_distance)

        # loss
        self.losses = self.p_score + self.margin - self.n_score
        self.loss = tf.reduce_sum(tf.clip_by_value(self.losses, clip_value_min=0., clip_value_max=1.))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer()

        # adjust gradient
        gradients = self.optimizer.compute_gradients(self.loss)
        gradients = [(self._to_riemannian_gradient(grad), var) for grad, var in gradients]
        self.train_op = self.optimizer.apply_gradients(gradients)

    def _to_riemannian_gradient(self, ge):
        if ge is None:
            return None
        try:
            shape = ge.get_shape().as_list()
            if len(shape) >= 3:
                grad_scale = 1 - tf.square(tf.norm(ge, axis=[-2, -1], keepdims=True))
            elif len(shape) == 2:
                grad_scale = 1 - tf.square(tf.norm(ge, keepdims=True))
            else:
                return ge
        except:
            grad_scale = 1 - tf.square(tf.norm(ge, keep_dims=True))

        grad_scale = (tf.square(grad_scale) + 1e-16) / 4.0
        gr = ge * grad_scale
        gr = tf.clip_by_norm(gr, 1.0, axes=0)
        return gr

    def _to_representation(self, sequence):
        embed = self.embedding(sequence)
        projected = self.projection(embed)
        rep = tf.reduce_sum(projected, 1, keepdims=True)
        return tf.clip_by_norm(rep, 1.0, axes=1)

    def _hyperbolic_distance(self, q, a, eps=1e-16):
        def _square_norm(x):
            return tf.square(tf.norm(x, keepdims=True, axis=1))

        z = _square_norm(q - a)
        q1 = 1.0 - _square_norm(q)
        a1 = 1.0 - _square_norm(a)
        return tf.acosh(1.0 + 2.0 * z / (q1 * a1 + eps))  # to avoid zero division
