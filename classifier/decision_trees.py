""" Decision Tree Classifier """

import logging

from sklearn.tree import DecisionTreeClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg


def main():
    settings_path = get_settings_path_from_arg("decision_tree_classifier")
    settings = load_settings(settings_path)
    
    init_logger(settings.log.dir, "decision_tree_classifier", logging.DEBUG)

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

    for i in range(0, 20):
        clf_k = DecisionTreeClassifier(min_impurity_decrease=0.001*i)
        ms.add_classifier("k_{}".format(i), clf_k)

    ms.get_best_model(k=10, plot=True)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()