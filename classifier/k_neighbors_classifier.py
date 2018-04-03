""" K - Nearest Neighbhors Classifier """

import logging

from sklearn.neighbors import KNeighborsClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings

def main():
    settings = load_settings("settings_classifier.json")

    init_logger(settings.log.dir, "k_neighbors_classifier", logging.DEBUG)

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

    clf = KNeighborsClassifier(3)
    clf_2 = KNeighborsClassifier(6)
    ms = ModelSelector()
    ms.add_selector(ds)
    ms.add_classifier(clf)
    ms.add_classifier(clf_2)
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()