# coding: utf-8
import itertools
from typing import Callable, Tuple

import numpy as np


class QADataset(object):
    def __init__(self, questions: np.ndarray, answers: np.ndarray) -> None:
        assert questions.shape == answers.shape
        assert len(questions.shape) == 2
        self._questions, self._q_indices = np.unique(questions, return_inverse=True, axis=0)
        self._answers, self._a_indices = np.unique(answers, return_inverse=True, axis=0)

    def random_sampling(self, negative_sampling_rate=1.0, include_positive=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # sizes
        positive_sampling_size = len(self._q_indices)
        negative_sampling_size = int(positive_sampling_size * negative_sampling_rate)
        sampling_size = positive_sampling_size + negative_sampling_size

        # choose indices
        qi = np.random.choice(range(len(self._questions)), 2 * sampling_size)
        ai = np.random.choice(range(len(self._answers)), 2 * sampling_size)

        # choose unique pairs
        original_pair_indices = set(zip(self._q_indices, self._a_indices))
        candidate_pair_indices = set(zip(qi, ai))
        negative_pair_indices = np.array(list(candidate_pair_indices - original_pair_indices))
        negative_pair_indices = negative_pair_indices[np.random.choice(range(negative_pair_indices.shape[0]), size=negative_sampling_size, replace=False)]
        negative_q_indices = negative_pair_indices[:, 0]
        negative_a_indices = negative_pair_indices[:, 1]

        labels = np.zeros(len(negative_q_indices))
        return self._shuffle(*self._add_positive(negative_q_indices, negative_a_indices, labels, include_positive))

    def max_sampling(self, scoring_functions: Callable[[np.ndarray, np.ndarray], float], include_positive=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate scores
        scores = np.array([scoring_functions(x, y) for x, y in itertools.product(self._questions, self._answers)]).reshape([len(self._questions), -1])
        scores[self._q_indices, self._a_indices] = 0.0  # set the scores of positive pairs to 0.

        # set indices
        negative_q_indices = np.array(range(self._questions.shape[0]))
        negative_a_indices = np.argmax(scores, axis=1)
        labels = np.zeros(negative_q_indices.shape[0])
        return self._shuffle(*self._add_positive(negative_q_indices, negative_a_indices, labels, include_positive))

    def _add_positive(self, q_indices, a_indices, labels, include_positive: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if include_positive:
            labels = np.append(labels, np.ones(len(self._q_indices)))
            q_indices = np.append(q_indices, self._q_indices)
            a_indices = np.append(a_indices, self._a_indices)
        return q_indices, a_indices, labels

    def _shuffle(self, q_indices, a_indices, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.array(range(len(q_indices)))
        np.random.shuffle(indices)
        return self._questions[q_indices[indices]], self._answers[a_indices[indices]], labels[indices]
