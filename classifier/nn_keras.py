## Importing standard libraries
import logging

## Importing standard libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Importing sklearn libraries

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import train_test_split
# from sklearn.preprocessing import LabelEncoder

## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

## Read data from the CSV file

class NNKeras:
    def __init__(self, inp):
        self.model = Sequential()
        self.model.add(Dense(1500,input_dim=inp,  kernel_initializer='uniform', activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1500, activation='sigmoid'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(99, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

        # self.early_stopping = EarlyStopping(monitor='val_loss', patience=280)

    def fit(self, x_train, y_train):
        y_cat = to_categorical(y_train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=280)
        self.model.fit(x_train,y_cat,batch_size=192,
                        epochs=800 ,verbose=0, validation_split=0.1, callbacks=[early_stopping])

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

        

def main():
    # data = pd.read_csv('data/train.csv')
    # parent_data = data.copy()    ## Always a good idea to keep a copy of original data
    # ID = data.pop('id')

    # y = data.pop('species')
    # y = LabelEncoder().fit(y).transform(y)
    # print(y.shape)

    # X = StandardScaler().fit(data).transform(data)
    # print(X.shape)

    # # y_cat = y
    # y_cat = to_categorical(y)
    # print(y_cat.shape)


    # model = Sequential()
    # model.add(Dense(1450,input_dim=192,  kernel_initializer='uniform', activation='relu'))
    # model.add(Dropout(0.05))
    # model.add(Dense(1450, activation='sigmoid'))
    # model.add(Dropout(0.05))
    # model.add(Dense(99, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=280)
    # history = model.fit(X,y_cat,batch_size=192,
    #                     epochs=800 ,verbose=0, validation_split=0.1, callbacks=[early_stopping])
                        


    # print('val_acc: ',max(history.history['val_acc']))
    # print('val_loss: ',min(history.history['val_loss']))
    # print('train_acc: ',max(history.history['acc']))
    # print('train_loss: ',min(history.history['loss']))

    # print()
    # print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

    # test = pd.read_csv('data/test.csv')
    # index = test.pop('id')
    # test = StandardScaler().fit(test).transform(test)
    # yPred = model.predict_proba(test)

    # yPred = pd.DataFrame(yPred,index=index,columns=sorted(parent_data.species.unique()))


    # fp = open('submission_nn_kernel.csv','w')
    # fp.write(yPred.to_csv())

    settings_path = get_settings_path_from_arg("k_neighbors_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "k_neighbors_classifier", logging.DEBUG)

    dl = DataLoader()
    dl.load_train(settings.data.train_path)
    dl.load_test(settings.data.test_path)
    dl.scale_data()
    # k = np.size(dl.classes)
    # dl.load_from_images(settings.data.image_path, k, k*3, verbose=False)

    # dr = DataReducer(dl.x_train, dl.x_test)
    # dr.pca_data_reduction()
    # dl.set_x_train(dr.get_pca_x_train())
    # dl.set_x_test(dr.get_pca_x_test())

    ms = ModelSelector()

    
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

    ms.add_selector("all_feature", ds)

    clf = NNKeras(ds.selected_x_test.shape[1])
    ms.add_classifier("nn_keras", clf)

    # ms.get_best_model(k=10)
    ms.best_classifier = clf
    ms.best_data_selector = ds
    ms.generate_submission(settings.data.submission_dir, dl.classes)


if __name__ == "__main__":
    main()