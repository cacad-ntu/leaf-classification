""" K - Nearest Neighbhors Classifier """

import logging

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def main():
    settings_path = get_settings_path_from_arg("k_neighbors_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "k_neighbors_classifier", logging.DEBUG)

    # Load test and training
    dl = DataLoader()
    dl.load_train(settings.data.train_path)
    dl.load_test(settings.data.test_path)
    dl.scale_data()

    # Image feature extraction
    k = np.size(dl.classes) *10
    dl.load_from_images(settings.data.image_path, k, k*3, verbose=False)

    # Dimensionality reduction
    dr = DataReducer(dl.x_train, dl.x_test)
    dr.pca_data_reduction()
    dl.set_x_train(dr.get_pca_x_train())
    dl.set_x_test(dr.get_pca_x_test())


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

    # clf_5 = KNeighborsClassifier(6, weights="distance")
    # ms.add_classifier("k_6_distance", clf_5)

    # clf_6 = KNeighborsClassifier(10, p=1)
    # ms.add_classifier("k_10_p1", clf_6)

    # clf_7 = KNeighborsClassifier(10, weights="distance", p=1)
    # ms.add_classifier("k_10_distance_p1", clf_7)

    # for i in range(1,20):
    #     clf_k = KNeighborsClassifier(i)
    #     ms.add_classifier("k_{}".format(i), clf_k)

    # for i in range(1,20):
    #     clf_k = KNeighborsClassifier(i, p=1)
    #     ms.add_classifier("k_{}_p1".format(i), clf_k)

    for i in range(1, 40):
        clf_k = KNeighborsClassifier(i, weights="distance", p=1)
        ms.add_classifier("k_{}_distance_p1".format(i), clf_k)
    # clf = KNeighborsClassifier(6, weights="distance", p=1)
    # ms.add_classifier("k_6_distance_p1", clf)

    # clf_7 = KNeighborsClassifier(6, weights="distance", p=3)
    # ms.add_classifier("k_6_distance_p3", clf_7)

    # clf_5 = KNeighborsClassifier(20)
    # ms.add_classifier("k_20", clf_5)

    # Get best model
    ms.get_best_model(k=10, plot=True)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()