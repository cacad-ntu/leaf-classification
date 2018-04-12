import csv
import cv2
import logging
import numpy as np
import os
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

from utils_leaf_classification.data_loader import DataLoader
from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def get_feature(img_path, ids, species, verbose=False):
    # print(ids)
    # print(species)
    img_des = {}
    dico = []
    sift = cv2.xfeatures2d.SIFT_create()
    for leaf in ids:
        image_path = os.path.join(img_path, str(leaf)+".jpg")
        print(image_path)
        img = cv2.imread(image_path)
        kp, des = sift.detectAndCompute(img, None)
        img_des[leaf] = [kp, des]

        for d in des:
            dico.append(d)

    k = np.size(species) * 10

    batch_size = np.size(os.listdir(img_path)) * 3
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=verbose).fit(dico)

    kmeans.verbose = verbose

    histo_column = ['id']

    for i in range(k):
        histo_column.append('image' + str(i))

    histo_list = pd.DataFrame(columns=histo_column)

    for leaf in ids:
        image_path = os.path.join(img_path, str(leaf)+".jpg")
        kp, des = img_des[leaf]

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

        # print(leaf)

        histo_df = pd.DataFrame([[leaf] + list(histo)], columns=histo_column)
        histo_list = histo_list.append(histo_df)
    
    return histo_list

if __name__ == "__main__":
    settings_path = get_settings_path_from_arg("image_extractor")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "image_extractor", logging.DEBUG)

    dl = DataLoader()
    dl.load_test(settings.data.test_path)
    dl.load_train(settings.data.train_path)

    print(get_feature(settings.data.image_path, dl.id_test, dl.classes))