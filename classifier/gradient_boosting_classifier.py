""" Gradient Boosting Classifier """

import logging

from sklearn.ensemble import GradientBoostingClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def main():
    settings_path = get_settings_path_from_arg("gradient_boosting_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "gradient_boosting_classifier", logging.DEBUG)
    ms = ModelSelector()

    # Load test and training
    dl = DataLoader()
    dl.load_train(settings.data.train_path)
    dl.load_test(settings.data.test_path)

    # Add Data Selector
    ds = DataSelector(
        dl.id_train, dl.x_train, dl.y_train,
        dl.id_test, dl.x_test
    )
    ds.add_all()

    # Add data selection to model selector
    ms.add_selector("all_feature", ds)

    # Add Classifier
    clf = GradientBoostingClassifier(n_estimators=100, random_state=0, min_samples_leaf=3, verbose=True)
    ms.add_classifier("gradient_boosting_classifier", clf)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()