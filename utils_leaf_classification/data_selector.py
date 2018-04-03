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

    def __init__(self, id=pd.DataFrame(), y=pd.DataFrame(), x=pd.DataFrame(), init_selected=False):
        self.id = id
        self.x = x
        self.y = y
        self.selected_x = pd.DataFrame()
        if init_selected:
            self.selected_x = x
        logging.info("[DataSelector] Initiate DataSelector(id={}, y={}, x={}, init_selected={})".format(
            id.shape, y.shape, x.shape, init_selected
        ))

    def set_x(self, x):
        """ set x values """
        logging.debug("[DataSelector] Setting x values")
        self.x = x

    def set_y(self, y):
        """ set y values """
        logging.debug("[DataSelector] Setting y values")
        self.y = y

    def set_id(self, id):
        """ set id values """
        logging.debug("[DataSelector] Setting id values")
        self.id = id

    def get_all_feature(self):
        """ Get all feature from data """
        return self.x.columns

    def get_selected_feature(self):
        """ get selected feature """
        return self.selected_x.columns

    def get_selected_x(self):
        """ Return dataframe with the selected feature """
        return self.selected_x

    def get_y(self):
        """ get all y """
        return self.y

    def get_id(self):
        """ get all id """
        return self.id

    def get_stratified_k_fold_data(self, n=10, id=True, y=True):
        """ get stratified fold of the data (iterator) """
        logging.debug("[DataSelector] Generating k fold data, n={}, id={}, y={}".format(n, id, y))
        skf = StratifiedKFold(n_splits=n)
        for train_idx, test_idx in skf.split(self.selected_x, self.y):
            train_x, test_x = self.selected_x.values[train_idx], self.selected_x.values[test_idx]
            if id and y:
                train_id, test_id = self.id[train_idx], self.id[test_idx]
                train_y, test_y = self.y[train_idx], self.y[test_idx]
                yield train_id, train_y, train_x, test_id, test_y, test_x
            elif id:
                train_id, test_id = self.id[train_idx], self.id[test_idx]
                yield train_id, train_x, test_id, test_x
            elif y:
                train_y, test_y = self.y[train_idx], self.y[test_idx]
                yield train_y, train_x, test_y, test_x
            else:
                yield train_x, test_x

    def add(self, feature, idx):
        """ add feature-idx to the selected data """
        logging.debug("[DataSelector] Adding feature {}{}".format(feature, idx))
        col_str = feature+str(idx)
        if (col_str not in self.selected_x.columns) and (col_str in self.x.columns):
            self.selected_x.insert(len(self.selected_x.columns), column=col_str, value=self.x[col_str])

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
        for item in self.x.columns:
            if feature in item:
                if item not in self.selected_x.columns:
                    self.selected_x.insert(len(self.selected_x.columns), column=item, value=self.x[item])

    def remove(self, feature, idx):
        """ remove feature-idx from the selected data """
        logging.debug("[DataSelector] Removing feature {}{}".format(feature, idx))
        col_str = feature+str(idx)
        if col_str in self.selected_x.columns:
            self.selected_x.drop(col_str, 1, inplace=True)

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
        for item in self.selected_x.columns:
            self.selected_x.drop(item, 1, inplace=True)

