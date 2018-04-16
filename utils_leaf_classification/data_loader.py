""" Load train and test data """

import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from utils_leaf_classification import image_feature_extractor


class DataLoader:
    """ Data loader class """

    def __init__(self, id_train=pd.DataFrame(), x_train=pd.DataFrame(), y_train=pd.DataFrame(), id_test=pd.DataFrame(), x_test=pd.DataFrame()):
        self.classes = None
        self.x_train = x_train
        self.y_train = y_train
        self.id_train = id_train
        self.x_test = x_test
        self.id_test = id_test
        self.scaler = None
        logging.info("[DataLoader] Initiate DataLoader(id_train={}, x_train={}, y_train={}, id_test={}, x_test={})".format(
            id_train.shape, x_train.shape, y_train.shape, id_test.shape, x_test.shape
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

    def load_from_images(self, image_path, k=None, batch_size=None, verbose=False):
        """ Load train and test feature from images """
        logging.info("[DataLoader] Loading data from image, path:{}".format(image_path))
        if not k:
            k = np.size(self.classes)

        all_id = self.id_train.append(self.id_test)

        all_histo_feature = image_feature_extractor.get_feature(image_path, all_id, k, batch_size, verbose)
        train_feature = self.id_train.to_frame().join(all_histo_feature.set_index('id'), on='id')
        self.x_train = self.x_train.join(train_feature.drop(['id'], axis=1))
        test_feature = self.id_test.to_frame().join(all_histo_feature.set_index('id'), on='id')
        self.x_test = self.x_test.join(test_feature.drop(['id'], axis=1))

    def scale_data(self):
        """ Scale test and train data """
        logging.info("[DataLoader] Scaling data")
        self.scaler = StandardScaler().fit(self.x_train)
        tmp_x_train = self.scaler.transform(self.x_train)
        for i in range(len(self.x_train.columns)):
            self.x_train[self.x_train.columns[i]] = tmp_x_train[:,[i]]
        tmp_x_test = self.scaler.transform(self.x_test)
        for i in range(len(self.x_test.columns)):
            self.x_test[self.x_test.columns[i]] = tmp_x_test[:,[i]]

    def set_x_train(self, x_train):
        """ Set x_train value """
        logging.debug("[DataLoader] Setting x train values {}".format(x_train.shape))
        self.x_train = x_train

    def set_x_test(self, x_test):
        """ Set x_test value """
        logging.debug("[DataLoader] Setting x test values {}".format(x_test.shape))
        self.x_test = x_test

    def get_train(self):
        """ get train data (id, x_train, y_train) """
        return self.id_train, self.x_train, self.y_train

    def get_test(self):
        """ get test data (id, x_test) """
        return self.id_test, self.x_test

    def get_class(self):
        """ get classes from data """
        return self.classes
