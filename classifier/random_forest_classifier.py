""" Random Forest Classifier """

import logging

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg


def main():
    settings_path = get_settings_path_from_arg("random_forest_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "random_forest_classifier", logging.DEBUG)
    ms = ModelSelector()

    # Load test and training
    dl = DataLoader()
    dl.load_train(settings.data.train_path)
    dl.load_test(settings.data.test_path)
    dl.scale_data()

    # Image feature extraction
    k = np.size(dl.classes) *10
    dl.load_from_images(settings.data.image_path, k, k*3, verbose=False)

    # Add Data Selector
    ds = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds.add_all()

    # Use lasso
    ds.auto_remove_lasso(0.17)

    # Dimensionality reduction
    dr = DataReducer(ds.train_x, ds.test_x)
    dr.pca_data_reduction()
    ds = DataSelector(
        dl.id_train, dr.x_train, dl.y_train,
        dl.id_test, dr.x_test
    )
    ds.add_all()

    # Add data selection to model selector
    ms.add_selector("all_feature", ds)

    # Add Classifier to model selector
    clf = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3, bootstrap=False)
    ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3, bootsrap=False", clf)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()