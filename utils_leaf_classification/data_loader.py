""" Load train and test data """

import logging

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class DataLoader:
    """ Data loader class """

    def __init__(self, id_train=None, x_train=None, y_train=None, id_test=None, x_test=None):
        self.classes = None
        self.x_train = x_train
        self.y_train = y_train
        self.id_train = id_train
        self.x_test = x_test
        self.id_test = id_test
        logging.info("[DataLoader] Initiate DataLoader(id_train={}, x_train={}, y_train={}, id_test={}, x_test={})".format(
            id_train, x_train, y_train, id_test, x_test
        ))

    def load_train(self, csv_file):
        """ load and preprocess train data from csv file """
        logging.info("[DataLoader] Loading train data from {}".format(csv_file))
        train_data = pd.read_csv(csv_file)
        label_encode = LabelEncoder().fit(train_data.species)
        self.id_train = train_data.id
        self.y_train = label_encode.transform(train_data.species)
        self.x_train = train_data.drop(['id', 'species'], axis=1)
        self.classes = list(label_encode.classes_)
        logging.info("[DataLoader] Train data successfully loaded from {}".format(csv_file))

    def load_test(self, csv_file):
        """ load and preprocess test data from csv file """
        logging.info("[DataLoader] Loading test data from {}".format(csv_file))
        test_data = pd.read_csv(csv_file)
        self.id_test = test_data.id
        self.x_test = test_data.drop(['id'], axis=1)
        logging.info("[DataLoader] Test data successfully loaded from {}".format(csv_file))

    def get_train(self):
        """ get train data (id, x_train, y_train) """
        return self.id_train, self.x_train, self.y_train

    def get_test(self):
        """ get test data (id, x_test) """
        return self.id_test, self.x_test

    def get_class(self):
        """ get classes from data """
        return self.classes
