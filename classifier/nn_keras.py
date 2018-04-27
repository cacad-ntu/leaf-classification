""" Neural Network Classifier """
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.data_reducer import DataReducer
from utils_leaf_classification.data_selector import DataSelector
from utils_leaf_classification.k_fold import ModelSelector
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

class NNKeras:
    def __init__(self, inp):
        self.model = Sequential()
        self.model.add(Dense(1500,input_dim=inp,  kernel_initializer='uniform', activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1500, activation='sigmoid'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(99, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

    def fit(self, x_train, y_train):
        y_cat = to_categorical(y_train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=280)
        self.model.fit(x_train,y_cat,batch_size=192,
                        epochs=800 ,verbose=0, validation_split=0.1, callbacks=[early_stopping])

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

        

def main():

    settings_path = get_settings_path_from_arg("k_neighbors_classifier")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "k_neighbors_classifier", logging.DEBUG)
    ms = ModelSelector()

    # Load test and training
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

    # Add data selection to model selector
    ms.add_selector("all_feature", ds)

    # Add Classifier
    clf = NNKeras(ds.selected_x_test.shape[1])
    ms.add_classifier("nn_keras", clf)

    # Get best model
    ms.get_best_model(k=10)
    ms.generate_submission(settings.data.submission_dir, dl.classes)

if __name__ == "__main__":
    main()