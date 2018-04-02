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

    ds = DataSelector(id=dl.id_train, y=dl.y_train, x=dl.x_train)
    ds.add('margin', 1)

    ds_test = DataSelector(id=dl.id_test, x=dl.x_test)
    ds_test.add('margin', 1)

    clf = KNeighborsClassifier(3)
    ms = ModelSelector()
    ms.set_data_selector(ds)
    ms.add_classifier(clf)
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes, ds_test.selected_x, ds_test.id)

if __name__ == "__main__":
    main()