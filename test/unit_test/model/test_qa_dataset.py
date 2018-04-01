# coding: utf-8
import unittest
import numpy as np
from model.qa_dataset import QADataset


class TestQADataset(unittest.TestCase):
    def test_random_sampling(self):
        questions = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        answers = np.array([[10, 1], [20, 2], [30, 3], [40, 4]])
        dataset = QADataset(questions=questions, answers=answers)
        original_pairs = list(zip(questions, answers))

        questions, answers, labels = dataset.random_sampling(negative_sampling_rate=1.0)
        for q, a, label in zip(questions, answers, labels):
            includes = self._includes_in_original(q, a, original_pairs)
            if label == 0:
                self.assertFalse(includes)
            else:
                self.assertTrue(includes)

    def test_max_sampling(self):
        questions = np.array([1, 2, 3, 4]).reshape([4, -1])
        answers = np.array([1.1, 2, 3.1, 1]).reshape([4, -1])
        dataset = QADataset(questions=questions, answers=answers)
        negative_questions, negative_answers, negative_labels = dataset.max_sampling(lambda x, y: abs(x - y), include_positive=False)
        expected_pairs = {(1, 3.1), (2, 3.1), (3, 1), (4, 1.1)}
        self.assertEqual(expected_pairs, set(zip(negative_questions.reshape(-1), negative_answers.reshape(-1))))

    def _includes_in_original(self, question, answer, original_pairs):
        for q, a in original_pairs:
            if np.array_equal(question, q) & np.array_equal(answer, a):
                return True
        return False


if __name__ == '__main__':
    unittest.main()


