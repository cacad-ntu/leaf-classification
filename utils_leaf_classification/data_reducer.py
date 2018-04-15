""" reduce data dimentionality """

import logging
import pandas as pd

from sklearn.decomposition import PCA

from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

class DataReducer:
    """ DataReducer class """

    def __init__(self, x_train=None, x_test=None):
        self.x_train = x_train
        self.x_test = x_test
        self.pca_x_train = None
        self.pca_x_test = None
        logging.info("[DataReducer] Initiate DataReducer(x_train={}, x_test={})".format(
            x_train.shape, x_test.shape
        ))

    def set_x_train(self, x_train):
        """ Set x_train value """
        logging.debug("[DataReducer] Setting x train values {}".format(x_train.shape))
        self.x_train = x_train

    def set_x_test(self, x_test):
        """ Set x_test value """
        logging.debug("[DataReducer] Setting x test values {}".format(x_test.shape))
        self.x_test = x_test

    def pca_data_reduction(self):
        """ reduce data dimension by PCA """
        logging.info("[DataReducer] Reducing data dimension")
        data = self.x_train.append(self.x_test)
        pca = PCA(n_components="mle", svd_solver="full")
        data_pca = pca.fit_transform(data)
        logging.info("[DataReducer] reduced data dimension: {}".format(data_pca.shape))
        logging.debug("[DataReducer] reduced data: {}".format(data_pca))

        col_names = []
        for i in range(data_pca.shape[1]):
            col_names.append("pca{}".format(i))

        self.pca_x_train = data_pca[:len(self.x_train)]
        self.pca_x_train = pd.DataFrame(self.pca_x_train, columns=col_names)
        self.pca_x_test = data_pca[len(self.x_train):]
        self.pca_x_test = pd.DataFrame(self.pca_x_test, columns=col_names)

    def get_pca_x_train(self):
        """ get x_train after pca """
        return self.pca_x_train

    def get_pca_x_test(self):
        """ get x_test after pca """
        return self.pca_x_test

if __name__ == "__main__":
    settings_path = get_settings_path_from_arg("data_reducer")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "data_reducer", logging.DEBUG)

    train_data = pd.read_csv(settings.data.train_path)
    test_data = pd.read_csv(settings.data.test_path)

    train_data = train_data.drop("species", axis=1)

    dr = DataReducer(train_data, test_data)
    dr.pca_data_reduction()

    pca_train_data = dr.get_pca_x_train()
    pca_test_data = dr.get_pca_x_test()

    print(pca_train_data.shape)
    print(pca_test_data)
