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

    ms = ModelSelector()

    # Add Data Selector
    ds = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds.add_all()
    ms.add_selector("all_feature", ds)

    # ds2 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds2.add_all('margin')
    # ms.add_selector("margin_only", ds)

    # ds3 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds3.add_all('shape')
    # ms.add_selector("shape_only", ds)

    # ds4 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds4.add_all('texture')
    # ms.add_selector("texture_only", ds)

    # ds5 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds5.add_all('texture')
    # ds5.add_all('shape')
    # ms.add_selector("texture_shape", ds5)

    # ds6 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds6.add_all('texture')
    # ds6.add_all('margin')
    # ms.add_selector("texture_margin", ds6)

    # ds7 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds7.add_all('margin')
    # ds7.add_all('shape')
    # ms.add_selector("margin_shape", ds7)

    # ds8 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds8.add_range('margin',0,33)
    # ds8.add_all('shape')
    # ds8.add_range('texture',33,65)
    # ms.add_selector("margin0-32 and shape and texture33-64", ds8)

    # Add Classifier
    # clf = KNeighborsClassifier(2)
    # ms.add_classifier("k_2", clf)

    # clf_2 = KNeighborsClassifier(4)
    # ms.add_classifier("k_4", clf_2)

    # clf_3 = KNeighborsClassifier(6)
    # ms.add_classifier("k_6", clf_3)

    # clf_4 = KNeighborsClassifier(10)
    # ms.add_classifier("k_10", clf_4)

    # clf_5 = KNeighborsClassifier(10, weights="distance")
    # ms.add_classifier("k_10_distance", clf_5)

    # clf_6 = KNeighborsClassifier(10, p=1)
    # ms.add_classifier("k_10_p1", clf_6)

    # clf_7 = KNeighborsClassifier(10, weights="distance", p=1)
    # ms.add_classifier("k_10_distance_p1", clf_7)

    # for i in range(10,20):
    #     clf_k = KNeighborsClassifier(i, weights="distance", p=1)
    #     ms.add_classifier("k_{}_distance_p1".format(i), clf_k)
    clf_11 = KNeighborsClassifier(11, weights="distance", p=1)
    ms.add_classifier("k_11_distance_p1", clf_11)

    # clf_7 = KNeighborsClassifier(10, weights="distance", p=3)
    # ms.add_classifier("k_10_distance_p3", clf_7)

    # clf_5 = KNeighborsClassifier(20)
    # ms.add_classifier("k_20", clf_5)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()