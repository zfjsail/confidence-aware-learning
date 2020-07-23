from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from collections import defaultdict as dd
import numpy as np
from os.path import join
import sklearn
import json
import codecs
import _pickle
import nltk
from torch.utils.data import Dataset, DataLoader

from utils import settings
from utils import feature_utils
from utils import data_utils


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class VenueCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle):

        self.file_dir = file_dir

        self.matrix_size_1_long = matrix_size1
        self.matrix_size_2_short = matrix_size2

        self.train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r'))
        self.mag = [nltk.word_tokenize(p[1]) for p in self.train_data]
        self.aminer = [nltk.word_tokenize(p[2]) for p in self.train_data]
        self.labels = [p[0] for p in self.train_data]

        self.calc_keyword_seqs()

        n_matrix = len(self.labels)
        self.X_long = np.zeros((n_matrix, self.matrix_size_1_long, self.matrix_size_1_long))
        self.X_short = np.zeros((n_matrix, self.matrix_size_2_short, self.matrix_size_2_short))
        self.Y = np.zeros(n_matrix, dtype=np.long)
        count = 0
        for i, cur_y in enumerate(self.labels):
            if i % 100 == 0:
                logger.info('pairs to matrices %d', i)
            v1 = self.mag[i]
            v1 = " ".join([str(v) for v in v1])
            v2 = self.aminer[i]
            v2 = " ".join([str(v) for v in v2])
            v1_key = self.mag_venue_keywords[i]
            v1_key = " ".join([str(v) for v in v1_key])
            v2_key = self.aminer_venue_keywords[i]
            v2_key = " ".join([str(v) for v in v2_key])
            matrix1 = self.sentences_long_to_matrix(v1, v2)
            self.X_long[count] = feature_utils.scale_matrix(matrix1)
            matrix2 = self.sentences_short_to_matrix(v1_key, v2_key)
            self.X_short[count] = feature_utils.scale_matrix(matrix2)
            self.Y[count] = cur_y
            count += 1

        print("shuffle", shuffle)
        if shuffle:
            self.X_long, self.X_short, self.Y = sklearn.utils.shuffle(
                self.X_long, self.X_short, self.Y,
                random_state=seed
            )

        self.labels_one_hot = data_utils.one_hot_encoding(self.Y)

        self.N = len(self.Y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_long[idx], self.X_short[idx], self.Y[idx], self.labels_one_hot[idx], idx

    def sentences_long_to_matrix(self, title1, title2):
        twords1 = feature_utils.get_words(title1)[: self.matrix_size_1_long]
        twords2 = feature_utils.get_words(title2)[: self.matrix_size_1_long]

        matrix = -np.ones((self.matrix_size_1_long, self.matrix_size_1_long))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def sentences_short_to_matrix(self, title1, title2):
        twords1 = feature_utils.get_words(title1)[: self.matrix_size_2_short]
        twords2 = feature_utils.get_words(title2)[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def calc_keyword_seqs(self):
        N = len(self.mag)
        mag_keywords = []
        aminer_keywords = []
        for i in range(N):
            cur_v_mag = self.mag[i]
            cur_v_aminer = self.aminer[i]
            overlap = set(cur_v_mag).intersection(cur_v_aminer)
            new_seq_mag = []
            new_seq_aminer = []
            for w in cur_v_mag:
                if w in overlap:
                    new_seq_mag.append(w)
            for w in cur_v_aminer:
                if w in overlap:
                    new_seq_aminer.append(w)
            mag_keywords.append(new_seq_mag)
            aminer_keywords.append(new_seq_aminer)
        self.mag_venue_keywords = mag_keywords
        self.aminer_venue_keywords = aminer_keywords
