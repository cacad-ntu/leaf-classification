""" Random Forest Classifier """

import logging

from sklearn.ensemble import RandomForestClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings

def main():
    settings = load_settings("settings_classifier.json")

    init_logger(settings.log.dir, "random_forest_classifier", logging.DEBUG)

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

    # ds2 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds2.add_all('margin')
    # ms.add_selector("margin_only", ds2)

    # ds3 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds3.add_all('shape')
    # ms.add_selector("shape_only", ds3)

    # ds4 = DataSelector(
    #     dl.id_train, dl.x_train, dl.y_train,
    #     dl.id_test, dl.x_test
    # )
    # ds4.add_all('texture')
    # ms.add_selector("texture_only", ds4)

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
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    # ms.add_classifier("max_depth=2, random_state=0", clf)

    # for i in range(5,15):
    #     clf_k = RandomForestClassifier(n_estimators=i, max_depth=2, random_state=0)
    #     ms.add_classifier("n_estimator={} ,max_depth=2, random_state=0".format(i), clf_k)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0)
    # ms.add_classifier("n_estimators=13, random_state=0", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, max_features='sqrt')
    # ms.add_classifier("n_estimators=13, random_state=0, max_features='sqrt'", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, max_features='log2')
    # ms.add_classifier("n_estimators=13, random_state=0, max_features='log2'", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, max_features=None)
    # ms.add_classifier("n_estimators=13, random_state=0, max_features=None", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_split=0.5)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_split=1", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_split=4)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_split=4", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=2)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=2", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=4)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=4", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=8)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=8", clf_2)

    # clf_2 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=16)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=16", clf_2)

    # clf_3 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3", clf_3)

    clf_4 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3, bootstrap=False)
    ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3, bootsrap=False", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3, oob_score=True)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3, oob_score=True", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, random_state=1, min_samples_leaf=3, bootstrap=False)
    # ms.add_classifier("n_estimators=13, random_state=1, min_samples_leaf=3, bootsrap=False", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, random_state=2, min_samples_leaf=3, bootstrap=False)
    # ms.add_classifier("n_estimators=13, random_state=2, min_samples_leaf=3, bootsrap=False", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3, bootstrap=False, warm_start=True)
    # ms.add_classifier("n_estimators=13, random_state=0, min_samples_leaf=3, bootsrap=False, warm_start=True", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, max_leaf_nodes=8)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, max_leaf_nodes=8", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, max_leaf_nodes=2)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, max_leaf_nodes=2", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, max_leaf_nodes=4)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, max_leaf_nodes=4", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, warm_start=True, min_impurity_decrease=0.000001)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, min_impurity_decrease=0.000001", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, warm_start=True, min_impurity_decrease=0.0001)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, min_impurity_decrease0.0001", clf_4)

    # clf_4 = RandomForestClassifier(n_estimators=13, min_samples_leaf=3, bootstrap=False, warm_start=True, min_impurity_decrease=0.01)
    # ms.add_classifier("n_estimators=13, min_samples_leaf=3, bootsrap=False, warm_start=True, min_impurity_decrease=0.01", clf_4)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()