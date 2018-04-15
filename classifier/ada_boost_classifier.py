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
from utils_leaf_classification.utility import init_logger, load_settings,get_settings_path_from_arg

def main():
    settings_path = get_settings_path_from_arg("ada_boost_classifier")
    settings = load_settings(settings_path)
    
    init_logger(settings.log.dir, "ada_boost_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_train(settings.data.train_path)
    dl.load_test(settings.data.test_path)

    ms = ModelSelector()

    # Add Data Selector
    ds = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds.add_all()
    ms.add_selector("all_feature", ds)

    ds2 = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds2.add_all("margin")
    ms.add_selector("margin_only", ds2)

    
    ds3 = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds3.add_all("shape")
    ms.add_selector("shape_only", ds3)


    ds4 = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds4.add_all("texture")
    ms.add_selector("texture_only", ds4)

    clf_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), algorithm='SAMME',learning_rate=0.01)
    clf_3 = AdaBoostClassifier(base_estimator=RandomForestClassifier(), algorithm='SAMME',learning_rate=0.01)
    clf_6 = AdaBoostClassifier(base_estimator=GaussianNB(), algorithm='SAMME',learning_rate=0.01)
    clf_7 = AdaBoostClassifier(learning_rate=0.01)
    ms.add_classifier("ada_decisiontrees_1",clf_1)
    ms.add_classifier("ada_randomforest_1",clf_3)
    ms.add_classifier("ada_gaussiannb",clf_6)
    ms.add_classifier("ada_test7",clf_7)
    ms.get_best_model(k=10, plot=True)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()