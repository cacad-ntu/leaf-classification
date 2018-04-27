""" Quadratic Discriminant Analysis Classifier """

import logging

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def main():
    settings_path = get_settings_path_from_arg("qda_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "qda_classifier", logging.DEBUG)
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

    # Add Classifier to model selector
    clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    ms.add_classifier("store_covariance=True", clf)

    clf = QuadraticDiscriminantAnalysis(store_covariance=False)
    ms.add_classifier("store_covariance=False", clf)

    clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False,
                              store_covariances=None, tol=0.0001)
    ms.add_classifier("final", clf)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()