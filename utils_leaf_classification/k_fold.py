""" K fold best model selector """

import logging
import os

import pandas as pd
from sklearn.metrics import log_loss

from utils_leaf_classification.utility import ensure_dir, get_now_str

class ModelSelector:
    """ Model selector class """

    def __init__(self, classifiers=[], data_selector=None, k=10):
        self.classifiers = classifiers
        self.data_selector = data_selector
        self.k = k
        self.best_classifier = None
        self.best_x_train = None
        self.best_y_train = None
        logging.info("[ModelSelector] Initiate ModelSelector(classifier={}, data_selector={}, k={})".format(
            classifiers, data_selector, k
        ))

    def set_data_selector(self, data_selector):
        """ set new data selector """
        logging.debug("[ModelSelector] Setting data_selector : {}".format(data_selector))
        self.data_selector = data_selector

    def set_fold(self, k):
        """ set number of fold """
        logging.debug("[ModelSelector] Setting k : {}".format(k))
        self.k = k

    def add_classifier(self, classifier):
        """ add classifier """
        logging.debug("[ModelSelector] Adding classifier {}".format(classifier.__class__.__name__))
        self.classifiers.append(classifier)

    def get_best_model(self, k=None):
        """ Select and return best model (classifier + train data) """
        logging.info("[ModelSelector] Selecting best model")
        if k:
            self.set_fold(k)

        best_log_loss = -1

        for classifier in self.classifiers:
            for y_train, x_train, y_test, x_test in self.data_selector.get_stratified_k_fold_data(n=self.k, id=False):
                name = classifier.__class__.__name__
                classifier.fit(x_train, y_train)
                y_predict = classifier.predict_proba(x_test)
                cur_log_loss = log_loss(y_test, y_predict)
                print("="*80)
                print(name)
                print("Log Loss: {}".format(cur_log_loss))
                logging.info("[ModelSelector] Testing {} with logloss:{}".format(name, cur_log_loss))
                if (best_log_loss < 0) or (cur_log_loss < best_log_loss):
                    best_log_loss = cur_log_loss
                    self.best_classifier = classifier
                    self.best_x_train = x_train
                    self.best_y_train = y_train

    def generate_submission(self, submission_dir, classes, x_test, id_test, ret=False):
        """ Generate submission csv """
        logging.info("[ModelSelector] Generating submission file")
        if not self.best_classifier:
            logging.error("Generating submission when best classifier is not set")
            raise ValueError("Generating submission when best classifier is not set")

        self.best_classifier.fit(self.best_x_train, self.best_y_train)
        predictions = self.best_classifier.predict_proba(x_test)

        submission = pd.DataFrame(predictions, columns=classes)
        submission.insert(0, 'id', id_test)
        submission.reset_index()

        ensure_dir(submission_dir)
        file_name = "Submission_" + get_now_str()+".csv"
        submission_file = os.path.join(submission_dir, file_name)
        submission.to_csv(submission_file, index = False)
