import logging
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

class DataReductor:
    def __init__(self, x_train = None, x_test = None):
        self.x_train = x_train
        self.x_test = x_test
        self.pca_x_train = None
        self.pca_x_test = None
        
    def pca_data_reduction(self):
        data = self.x_train.append(self.x_test)
        pca = PCA(n_components="mle", svd_solver="full")
        data_pca = pca.fit_transform(data)
        print(data_pca[len(self.x_train)])

        self.pca_x_train = data_pca[:len(self.x_train)]
        self.pca_x_test = data_pca[len(self.x_train):]

    def get_pca_x_train(self):
        return self.pca_x_train

    def get_pca_x_test(self):
        return self.pca_x_test

if __name__ == "__main__":
    train_data = pd.read_csv("..\\data\\train.csv")
    test_data = pd.read_csv("..\\data\\test.csv")

    train_data = train_data.drop("species", axis=1)

    all_data = train_data.append(test_data)

    dr = DataReductor(train_data, test_data)
    dr.pca_data_reduction()

    pca_train_data = dr.get_pca_x_train()

    pca_test_data = dr.get_pca_x_test()

    # print(pca_train_data[0])
    # print(pca_test_data[0])
