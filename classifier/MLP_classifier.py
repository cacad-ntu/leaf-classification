from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings

from sklearn.neural_network import MLPClassifier

import logging
import numpy
import pandas

def main():
    settings = load_settings("settings.json")

    init_logger(settings.log.dir, "MLP_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_test(settings.data.test_path)
    dl.load_train(settings.data.train_path)

    ds = DataSelector(dl.id_train, dl.x_train, dl.y_train, dl.id_test, dl.x_test)
    ds.add_all()

    #Available Parameter
    #hidden_layer_sizes(100,)
    #activation('relu') = 'identity', 'logistic', 'tanh', 'relu'
    #solver('adam') = 'lbfgs', 'sgd', 'adam'
    #alpha(1e-4)
    #batch_size(min(200, n_samples)) --> 'lbfgs' solver no batch size
    #learning_rate('constant') = 'constant', 'invscaling', 'adaptive' --> used at 'sgd' solver
    #learing_rate_init(1e-3) --> 'sgd' or 'adam' solver
    #power_t(0.5) --> used for 'invscaling' learning rate with 'sgd' solver
    #max_iter(200)
    #shuffle('True') --> when solver is 'sgd' or 'adam'
    #random_state
    #tol(1e-4) --> tolerance for optimisation
    #verbose('False')
    #warm_start('False') --> use previous solution
    #momentum(0.9) --> 'sgd' solver & momentum > 0
    #nestervos_momentum('True') 'sgd' solver & momentum > 0
    #early_stopping('False') --> 'sgd' or 'adam' solver
    #validation_fraction(0.1) --> only used when early_stopping is 'True'
    #beta_1(0.9) --> 'adam' solver only
    #beta_2(0.999) --> 'adam' solver only
    #epsilon(1e-8) --> 'adam' solver only


    #margin only = 0.53 - 0.54
    #

    classifier = MLPClassifier(hidden_layer_sizes = (150), max_iter = 1000)

    ms = ModelSelector()
    ms.add_selector("ds_1", ds)

    ms.add_classifier("MLP1", classifier)

    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()


