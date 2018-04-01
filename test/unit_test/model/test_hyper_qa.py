import unittest
import numpy as np

from model.hyper_qa import HyperQA


class TestHyperQA(unittest.TestCase):
    def test_model(self):
        model = HyperQA(sequence_length=100, projection_dim=100, margin=1.0, embedding=np.ones([100, 20]))


if __name__ == '__main__':
    unittest.main()

