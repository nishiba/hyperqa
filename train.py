# coding: utf-8
import os
from typing import Dict, List

import pandas as pd
import numpy as np
import csv

from nltk import word_tokenize

from model.hyper_qa import HyperQA


def read_embedding():
    data = pd.read_table('./data/glove.6B.200d.txt', sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
    embedding = data.values
    dictionary = dict(zip(data.index.values, range(data.shape[0])))
    dictionary['eos'] = len(dictionary)
    return embedding, dictionary


def to_sequences(tests: List[str], dictionary, max_length):
    def _to_sequence(tokens):
        return np.array([dictionary.get(t) for t in tokens if t in dictionary])
    sequences = [_to_sequence(word_tokenize(t))[:max_length] for t in tests]
    sequences = [np.pad(s, (0, max_length - len(s)), mode='constant', constant_values=dictionary['eos']) for s in sequences]
    return sequences


def read_train_data(dictionary: Dict[str, int], max_length: int):
    data = pd.read_csv('./data/WikiQA-train.tsv', sep='\t')
    questions = data.Question.values
    answers = data.Sentence.values
    labels = data.Label.values
    questions = to_sequences(questions, dictionary, max_length)
    answers = to_sequences(answers, dictionary, max_length)
    return questions, answers, labels


def train():
    # prepare
    sequence_length = 100
    projection_dim = 200
    margin = 1.0
    embedding, dictionary = read_embedding()
    questions, answers, labels = read_train_data(dictionary, max_length=30)

    # train
    model = HyperQA(sequence_length=sequence_length, projection_dim=projection_dim, margin=margin, embedding=embedding)


if __name__ == '__main__':
    train()
