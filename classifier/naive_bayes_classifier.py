""" Naive Bayes Classifier """

import logging
import re
import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RandomizedLasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg
from utils_leaf_classification.image_feature_extractor import get_feature 

def main():
    settings_path = get_settings_path_from_arg("naive_bayes_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "naive_bayes_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_test(settings.data.test_path)
    dl.load_train(settings.data.train_path)
    dl.scale_data()
    # k = np.size(dl.classes) * 10
    # dl.load_from_images(settings.data.image_path, k, k*3, verbose=False)

    # dr = DataReducer(dl.x_train, dl.x_test)
    # dr.pca_data_reduction()
    # dl.set_x_train(dr.get_pca_x_train())
    # dl.set_x_test(dr.get_pca_x_test())

    # names = dl.x_train.columns.tolist()
    
    # rlasso = RandomizedLasso(alpha=0.025)
    # rlasso.fit(dl.x_train, dl.y_train)

    # result = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
    #              names), reverse=True)

    # lasso_threshold = 0.02

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

    # clf = MultinomialNB(alpha=0.000001)
    # clf_2 = BernoulliNB(alpha=0.000001, binarize=0.000001)
    # clf_2 = BernoulliNB(alpha=0.001, binarize=0.001)
    # clf_2 = BernoulliNB(alpha=1, binarize=0.001)
    # clf_2 = BernoulliNB(alpha=1, binarize=0.01)
    clf_2 = BernoulliNB(alpha=3.5, binarize=0.03)
    ms = ModelSelector()
    # for i in range(1,101):
    #     clf_2 = BernoulliNB(alpha=i*0.1, binarize=0.03)
    #     ms.add_classifier("bernoulli: alpha={}, binarize=0.03".format(i*0.1), clf_2)
    # ms.add_selector("all features", ds)
    # ms.add_selector("margins only", ds2)
    # ms.add_selector("shape only", ds3)
    # ms.add_selector("texture only", ds4)
    # ms.add_selector("margin_16_48", ds5)
    # ms.add_selector("shape_16_48", ds6)
    # ms.add_selector("texture_16_48", ds7)
    # ms.add_selector("combined_16_48", ds8)
    # ms.add_classifier("multinb", clf)
    ms.add_classifier("bernoulli", clf_2)
    # for i in range(1, 201):
    #     ds_lasso = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    #     ds_lasso.auto_add_lasso(0.005*i)
    #     ms.add_selector("lasso{}".format(i), ds_lasso)
    ds_lasso = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    ds_lasso.auto_add_lasso(0.17)
    ms.add_selector("lasso", ds_lasso)
    ms.get_best_model(k=10)
    # print(get_feature(settings.data.image_path, dl.id_test, dl.classes))
    # ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()