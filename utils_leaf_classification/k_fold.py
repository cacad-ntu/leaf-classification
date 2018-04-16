""" K fold best model selector """

import logging
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from utils_leaf_classification.utility import ensure_dir, get_now_str

class ModelSelector:
    """ Model selector class """

    def __init__(self, classifiers={}, data_selector={}, k=10):
        self.classifiers = classifiers
        self.data_selector = data_selector
        self.k = k
        self.best_classifier_key = None
        self.best_classifier = None
        self.best_data_selector_key = None
        self.best_data_selector = None
        self.best_y_train = None
        logging.info("[ModelSelector] Initiate ModelSelector(classifier={}, data_selector={}, k={})".format(
            classifiers.__class__.__name__, data_selector.__class__.__name__, k
        ))

    def add_selector(self, key, data_selector):
        """ Add data selector """
        logging.debug("[ModelSelector] Adding data_selector : {}".format(data_selector))
        self.data_selector[key] = data_selector

    def set_fold(self, k):
        """ set number of fold """
        logging.debug("[ModelSelector] Setting k : {}".format(k))
        self.k = k

    def add_classifier(self, key, classifier):
        """ add classifier """
        logging.debug("[ModelSelector] Adding classifier {}".format(classifier.__class__.__name__))
        self.classifiers[key] = classifier

    def get_best_model(self, k=None, plot=False):
        """ Select and return best model (classifier + train data) """
        logging.info("[ModelSelector] Selecting best model")
        if k:
            self.set_fold(k)

        best_log_loss = -1
        log_cols=["Model", "Log Loss"]
        list_of_log_loss = pd.DataFrame(columns=log_cols)

        for classifier_key, classifier in self.classifiers.items():
            for data_selector_key, data_selector in self.data_selector.items():
                name = classifier.__class__.__name__
                cur_log_loss = 0
                for y_train, x_train, y_test, x_test in data_selector.get_stratified_k_fold_data(n=self.k, id=False):
                    classifier.fit(x_train, y_train)
                    y_predict = classifier.predict_proba(x_test)
                    cur_log_loss += log_loss(y_test, y_predict)
                cur_log_loss = cur_log_loss/self.k
                log_entry = pd.DataFrame([["{}({})-{}".format(name, classifier_key, data_selector_key), cur_log_loss]], columns=log_cols)
                list_of_log_loss = list_of_log_loss.append(log_entry)
                print("="*80)
                print("Classifier: {} ({})".format(classifier_key, name))
                print("Data_Selector: {}".format(data_selector_key))
                print("Log Loss: {}".format(cur_log_loss))
                logging.info("[ModelSelector] Testing classifier:{} ({}), data_selector:{} with logloss:{}".format(classifier_key, name, data_selector_key, cur_log_loss))
                if (best_log_loss < 0) or (cur_log_loss < best_log_loss):
                    best_log_loss = cur_log_loss
                    self.best_classifier_key = classifier_key
                    self.best_classifier = classifier
                    self.best_data_selector_key = data_selector_key
                    self.best_data_selector = data_selector

        print("="*80)
        logging.info("[ModelSelector] Best Classifier: {} ({})".format(self.best_classifier_key, self.best_classifier.__class__.__name__))
        print("**Best Classifier**: {} ({})".format(self.best_classifier_key, self.best_classifier.__class__.__name__))
        logging.info("[ModelSelector] Best Data Selector: {}".format(self.best_data_selector_key))
        print("**Best Data Selector**: {}".format(self.best_data_selector_key))
        logging.info("[ModelSelector] Logloss: {}".format(best_log_loss))
        print("**Logloss**: {}".format(best_log_loss))
        if plot:
            sns.set_color_codes("muted")
            sns.barplot(x='Log Loss', y='Model', data=list_of_log_loss, color='r')

            plt.xlabel('Log Loss')
            plt.title('Classifier Log Loss')
            plt.show()

    def generate_submission(self, submission_dir, classes, classifier=None, ret=False, smoothing=-1):
        """ Generate submission csv """
        logging.info("[ModelSelector] Generating submission file")
        if classifier == None:
            if self.best_classifier == None:
                logging.error("Generating submission when best classifier is not set")
                raise ValueError("Generating submission when best classifier is not set")
            else:
                classifier = self.best_classifier

        classifier.fit(self.best_data_selector.selected_x_train, self.best_data_selector.train_y)
        predictions = classifier.predict_proba(self.best_data_selector.selected_x_test)

        if smoothing>0:
            logging.info("Applying threshold smoothing in predictions ...")
            for i in range(len(predictions)):
                max_value = max(predictions[i])
                if (max_value > smoothing) or (list(predictions[i]).count(max_value) != 1):
                    continue
                for j in range(len(predictions[i])):
                    if predictions[i][j] < max_value:
                        predictions[i][j] = 0
                    else:
                        predictions[i][j] = 1

        submission = pd.DataFrame(predictions, columns=classes)
        submission.insert(0, 'id', self.best_data_selector.test_id)
        submission.reset_index()

        ensure_dir(submission_dir)
        file_name = "Submission_" + get_now_str()+".csv"
        submission_file = os.path.join(submission_dir, file_name)
        submission.to_csv(submission_file, index = False)
