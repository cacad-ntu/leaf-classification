"""
Select Data feature
add feature
remove feature
"""

import logging

import pandas as pd

from sklearn.model_selection import StratifiedKFold


class DataSelector:
    """ Data selector class """

    def __init__(self, train_id=pd.DataFrame(), train_x=pd.DataFrame(), train_y=pd.DataFrame(), test_id=pd.DataFrame(), test_x=pd.DataFrame(), init_selected=False):
        self.train_id = train_id
        self.train_x = train_x
        self.train_y = train_y
        self.test_id = test_id
        self.test_x = test_x
        self.selected_x_train = pd.DataFrame()
        self.selected_x_test = pd.DataFrame()
        if init_selected:
            self.selected_x_train = train_x
            self.selected_x_test = test_x
        logging.info("[DataSelector] Initiate DataSelector(train_id={}, train_x={}, train_y={}, test_id={}, test_x={}, init_selected={})".format(
            train_id.shape, train_x.shape, train_y.shape, test_id.shape, train_x.shape, init_selected
        ))

    def set_x_train(self, x):
        """ set x train values """
        logging.debug("[DataSelector] Setting x train values {}".format(x.shape))
        self.train_x = x

    def set_x_test(self, x):
        """ set x test values """
        logging.debug("[DataSelector] Setting x test values {}".format(x.shape))
        self.test_x = x

    def set_train_id(self, ids):
        """ set train id values """
        logging.debug("[DataSelector] Setting train id values {}".format(ids.shape))
        self.train_id = ids

    def set_test_id(self, ids):
        """ set test id values """
        logging.debug("[DataSelector] Setting test id values {}".format(ids.shape))
        self.test_id = ids

    def set_y_train(self, y):
        """ set y train values """
        logging.debug("[DataSelector] Setting y train values {}".format(y.shape))
        self.train_y = y

    def get_all_feature(self):
        """ Get all feature from data """
        return self.train_x.columns

    def get_selected_feature(self):
        """ get selected feature """
        return self.selected_x_train.columns

    def get_selected_x(self):
        """ Return dataframe with the selected feature """
        return self.selected_x_train, self.selected_x_test

    def get_id(self):
        """ get all id """
        return self.train_id, self.test_id

    def get_stratified_k_fold_data(self, n=10, id=True, y=True):
        """ get stratified fold of the data (iterator) """
        logging.debug("[DataSelector] Generating k fold data, n={}, id={}, y={}".format(n, id, y))
        skf = StratifiedKFold(n_splits=n)
        for train_idx, test_idx in skf.split(self.selected_x_train, self.train_y):
            train_x, test_x = self.selected_x_train.values[train_idx], self.selected_x_train.values[test_idx]
            if id and y:
                train_id, test_id = self.train_id[train_idx], self.train_id[test_idx]
                train_y, test_y = self.train_y[train_idx], self.train_y[test_idx]
                yield train_id, train_y, train_x, test_id, test_y, test_x
            elif id:
                train_id, test_id = self.train_id[train_idx], self.train_id[test_idx]
                yield train_id, train_x, test_id, test_x
            elif y:
                train_y, test_y = self.train_y[train_idx], self.train_y[test_idx]
                yield train_y, train_x, test_y, test_x
            else:
                yield train_x, test_x

    def add(self, feature, idx):
        """ add feature-idx to the selected data """
        logging.debug("[DataSelector] Adding feature {}{}".format(feature, idx))
        col_str = feature+str(idx)
        if (col_str not in self.selected_x_train.columns) and (col_str in self.train_x.columns):
            self.selected_x_train.insert(len(self.selected_x_train.columns), column=col_str, value=self.train_x[col_str])
            self.selected_x_test.insert(len(self.selected_x_test.columns), column=col_str, value=self.test_x[col_str])

    def add_range(self, feature, idx_start, idx_end):
        """ add feature-n from idx_start to idx_end """
        logging.info("[DataSelector] Adding batch feature {} from {} to {}".format(feature, idx_start, idx_end))
        for idx in range(idx_start, idx_end):
            self.add(feature, idx)

    def add_array(self, feature, idx_array):
        """ add feature-n from idx_array """
        logging.info("[DataSelector] Adding batch feature {} from {}".format(feature, idx_array))
        for idx in idx_array:
            self.add(feature, idx)

    def add_all(self, feature=""):
        """ add all feature """
        logging.info("[DataSelector] Adding batch all feature {}".format(feature))
        for item in self.train_x.columns:
            if feature in item:
                if item not in self.selected_x_train.columns:
                    self.selected_x_train.insert(len(self.selected_x_train.columns), column=item, value=self.train_x[item])
                    self.selected_x_test.insert(len(self.selected_x_test.columns), column=item, value=self.test_x[item])

    def remove(self, feature, idx):
        """ remove feature-idx from the selected data """
        logging.debug("[DataSelector] Removing feature {}{}".format(feature, idx))
        col_str = feature+str(idx)
        if col_str in self.selected_x_train.columns:
            self.selected_x_train.drop(col_str, 1, inplace=True)
            self.selected_x_test.drop(col_str, 1, inplace=True)

    def remove_range(self, feature, idx_start, idx_end):
        """ remove feature-n from idx_start to idx_end """
        logging.info("[DataSelector] Removing batch feature {} from {} to {}".format(feature, idx_start, idx_end))
        for idx in range(idx_start, idx_end):
            self.remove(feature, idx)

    def remove_array(self, feature, idx_array):
        """ remove feature-n from idx_array """
        logging.info("[DataSelector] Removing batch feature {} from {}".format(feature, idx_array))
        for idx in idx_array:
            self.remove(feature, idx)

    def remove_all(self, feature):
        """ remove all feature """
        logging.info("[DataSelector] Removing batch all feature {}".format(feature))
        for item in self.selected_x_train.columns:
            if feature in item:
                self.selected_x_train.drop(item, 1, inplace=True)
                self.selected_x_test.drop(item, 1, inplace=True)

