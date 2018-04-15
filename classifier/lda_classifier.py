""" Linear Discriminant Analysis Classifier """

import logging
import re

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RandomizedLasso
from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def main():
    settings_path = get_settings_path_from_arg("lda_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "lda_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_test(settings.data.test_path)
    dl.load_train(settings.data.train_path)
    dl.scale_data()
    # names = dl.x_train.columns.tolist()
    
    # rlasso = RandomizedLasso(alpha=0.025)
    # rlasso.fit(dl.x_train, dl.y_train)

    # result = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
    #              names), reverse=True)

    ds_lasso = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    ds_lasso.auto_add_lasso(0.08)
    # lasso_threshold = 0.1

    # for i, j in result:
    #     r = re.compile('([a-zA-Z]+)([0-9]+)')
    #     feature = r.match(j).groups()[0]
    #     idx = r.match(j).groups()[1]
    #     print("Feature: {}, idx: {}".format(feature, idx))
    #     if i >= lasso_threshold:
    #         ds_lasso.add(feature, idx)

    # ds = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds.add_all()

    # ds2 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds2.add_all('margin')

    # ds3 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds3.add_all('shape')

    # ds4 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds4.add_all('texture')

    # ds5 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds5.add_range('margin', 16, 48)
    
    # ds6 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds6.add_range('shape', 16, 48)

    # ds7 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds7.add_range('texture', 16, 48)

    # ds8 = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    # ds8.add_range('margin', 16, 48)
    # ds8.add_range('shape', 16, 48)
    # ds8.add_range('texture', 16, 48)

    clf = LinearDiscriminantAnalysis()
    ms = ModelSelector()
    # ms.add_selector("all features", ds)
    # ms.add_selector("margins only", ds2)
    # ms.add_selector("shape only", ds3)
    # ms.add_selector("texture only", ds4)
    # ms.add_selector("margin_16_48", ds5)
    # ms.add_selector("shape_16_48", ds6)
    # ms.add_selector("texture_16_48", ds7)
    # ms.add_selector("combined_16_48", ds8)
    ms.add_selector("lasso", ds_lasso)
    ms.add_classifier("lda", clf)
    ms.get_best_model(k=10)
    # ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()