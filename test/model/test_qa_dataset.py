# coding: utf-8
import unittest
import numpy as np
from model.qa_dataset import QADataset


class TestQADataset(unittest.TestCase):
    def setUp(self):
        questions = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]
        answers = [np.array([10, 1]), np.array([20, 2]), np.array([30, 3]), np.array([40, 4])]
        self.dataset = QADataset(questions=questions, answers=answers)
        self.original_pairs = list(zip(questions, answers))

    def test_random_sampling(self):
        questions, answers, labels = self.dataset.random_sampling(negative_sampling_rate=1.0)
        for q, a, label in zip(questions, answers, labels):
            includes = self._includes(q, a)
            if label == 0:
                self.assertFalse(includes)
            else:
                self.assertTrue(includes)

    def _includes(self, question, answer):
        for q, a in self.original_pairs:
            if np.array_equal(question, q) & np.array_equal(answer, a):
                return True
        return False


if __name__ == '__main__':
    unittest.main()


