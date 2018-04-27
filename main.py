""" Main class of leaf classification """

import logging
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings,get_settings_path_from_arg
from classifier.nn_keras import NNKeras

def main():
    settings_path = get_settings_path_from_arg("main_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "main_classifier", logging.DEBUG)
    ms = ModelSelector()

    # Load test and training data
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

    # Instantiate all the classifiers to be added
    clf_knn = KNeighborsClassifier(6, weights="distance", p=1)
    clf_adaboost = AdaBoostClassifier(learning_rate=0.01)
    clf_dectree = DecisionTreeClassifier(min_impurity_decrease=0.02)
    clf_gaussian = GaussianNB()
    clf_lda = LinearDiscriminantAnalysis()
    clf_gradientboost = GradientBoostingClassifier(n_estimators=100, random_state=0, min_samples_leaf=3, verbose=True)
    clf_bernoullinb = BernoulliNB(alpha=3.5, binarize=0.03)
    clf_nnkeras = NNKeras(ds.selected_x_test.shape[1])
    clf_svc = SVC(probability=True, C=1000, gamma=1)
    clf_nusvc = NuSVC(nu=0.1, gamma=10, probability=True)
    clf_qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    clf_randomforest = RandomForestClassifier(n_estimators=13, random_state=0, min_samples_leaf=3, bootstrap=False)
    
    # Add all the classifiers to the model selector
    ms.add_classifier("KNN", clf_knn)
    ms.add_classifier("AdaBoost", clf_adaboost)
    ms.add_classifier("Decision Tree", clf_dectree)
    ms.add_classifier("Gaussian NB", clf_gaussian)
    ms.add_classifier("LDA", clf_lda)
    ms.add_classifier("Gradient Boosting", clf_gradientboost)
    ms.add_classifier("Bernoulli NB", clf_bernoullinb)
    ms.add_classifier("NN Keras", clf_nnkeras)
    ms.add_classifier("SVC", clf_svc)
    ms.add_classifier("NuSVC", clf_nusvc)
    ms.add_classifier("QDA", clf_qda)
    ms.add_classifier("Random Forest", clf_randomforest)

    # Get best model
    ms.get_best_model(k=10, plot=True)
    # ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()