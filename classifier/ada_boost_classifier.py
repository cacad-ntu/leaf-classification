""" Ada Boost Classifier """

import logging

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings

def main():
    settings = load_settings("settings_classifier.json")

    init_logger(settings.log.dir, "ada_boost_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_test(settings.data.test_path)
    dl.load_train(settings.data.train_path)

    ds = DataSelector()
    ds.set_train_id(dl.id_train)
    ds.set_x_train(dl.x_train)
    ds.set_y_train(dl.y_train)
    ds.set_test_id(dl.id_test)
    ds.set_x_test(dl.x_test)
    ds.add_all()

    ds2 = DataSelector()
    ds2.set_train_id(dl.id_train)
    ds2.set_x_train(dl.x_train)
    ds2.set_y_train(dl.y_train)
    ds2.set_test_id(dl.id_test)
    ds2.set_x_test(dl.x_test)
    ds2.add_all("margin")
    
    ds3 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    ds3.add_all("shape")

    ds4 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    ds4.add_all("texture")

    clf_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), algorithm='SAMME',learning_rate=0.01)
    clf_3 = AdaBoostClassifier(base_estimator=RandomForestClassifier(), algorithm='SAMME',learning_rate=0.01)
    clf_6 = AdaBoostClassifier(base_estimator=GaussianNB(), algorithm='SAMME',learning_rate=0.01)
    clf_7 = AdaBoostClassifier(learning_rate=0.01)
    ms = ModelSelector()
    ms.add_selector("all_features",ds)
    ms.add_selector("only_margin",ds2)
    ms.add_selector("only_shape",ds3)
    ms.add_selector("only_texture",ds4)
    ms.add_classifier("ada_test1",clf_1)
    ms.add_classifier("ada_test3",clf_3)
    ms.add_classifier("ada_test6",clf_6)
    ms.add_classifier("ada_test7",clf_7)
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()