# coding: utf-8
import numpy as np
from typing import List


class QADataset(object):
    def __init__(self, questions: List[np.ndarray], answers: List[np.ndarray]) -> None:
        assert len(questions) == len(answers)
        self.questions, self.q_indices = np.unique(questions, return_inverse=True, axis=0)
        self.answers, self.a_indices = np.unique(answers, return_inverse=True, axis=0)

    def random_sampling(self, negative_sampling_rate=1.0):
        # sizes
        positive_sampling_size = len(self.q_indices)
        negative_sampling_size = int(positive_sampling_size * negative_sampling_rate)
        sampling_size = positive_sampling_size + negative_sampling_size

        # choose indices
        qi = np.random.choice(range(len(self.questions)), sampling_size)
        ai = np.random.choice(range(len(self.answers)), sampling_size)

        # choose unique pairs
        original_pair_indices = set(zip(self.q_indices, self.a_indices))
        candidate_pair_indices = set(zip(qi, ai))
        new_pair_indices = np.array(list(candidate_pair_indices - original_pair_indices))
        new_pair_indices = new_pair_indices[np.random.choice(range(len(new_pair_indices)), size=negative_sampling_size, replace=False)]
        new_q_indices = new_pair_indices[:, 0]
        new_a_indices = new_pair_indices[:, 1]

        # merge
        labels = np.array([1] * len(self.q_indices) + [0] * len(new_q_indices))
        q = np.append(self.q_indices, new_q_indices)
        a = np.append(self.a_indices, new_a_indices)
        indices = np.array(range(len(q)))
        np.random.shuffle(indices)
        return self.questions[q[indices]], self.answers[a[indices]], labels[indices]

    def max_sampling(self):
        pass
