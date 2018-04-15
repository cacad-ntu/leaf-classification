""" Utility to extract features from images """

import csv
import cv2
import logging
import numpy as np
import os
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

from utils_leaf_classification.utility import init_logger, load_settings, get_settings_path_from_arg

def get_feature(img_path, ids, k, batch_size=None, verbose=False):
    """ Load features from image using bag of word with sift descriptors """
    logging.info("[ImageFeatureExtractor] Extracting feature from {}".format(img_path))
    logging.debug("[ImageFeatureExtractor] Extracting feature with ids:{}".format(ids))
    img_des = {}
    dico = []
    sift = cv2.xfeatures2d.SIFT_create()
    for leaf in ids:
        image_path = os.path.join(img_path, str(leaf)+".jpg")
        logging.debug("[ImageFeatureExtractor] Extracting from {}".format(image_path))
        img = cv2.imread(image_path)
        kp, des = sift.detectAndCompute(img, None)
        img_des[leaf] = [kp, des]

        for d in des:
            dico.append(d)

    logging.info("[ImageFeatureExtractor] Clustering descriptors of all images")
    if not batch_size:
        batch_size = k
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=verbose).fit(dico)

    logging.info("[ImageFeatureExtractor] Preparing histogram dataframe")
    histo_column = ['id']
    for i in range(k):
        histo_column.append('image' + str(i))

    histo_list = pd.DataFrame(columns=histo_column)

    logging.info("[ImageFeatureExtractor] Counting histogram for all images")
    for leaf in ids:
        logging.debug("[ImageFeatureExtractor] Histogram image:{}".format(leaf))
        image_path = os.path.join(img_path, str(leaf)+".jpg")
        kp, des = img_des[leaf]

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_df = pd.DataFrame([[leaf] + list(histo)], columns=histo_column)
        histo_list = histo_list.append(histo_df)

    logging.info("[ImageFeatureExtractor] Histogram generated")
    logging.debug("[ImageFeatureExtractor] Histogram:{}".format(histo_list))
    return histo_list

def extract_descriptor_to_csv(img_path):
    """Extract image descriptor into csv"""

def extract_feature_to_csv(img_path, ids, k, batch_size=None, verbose=False):
    """Extract feature from image and save to csv"""

if __name__ == "__main__":
    settings_path = get_settings_path_from_arg("image_feature_extractor")
    settings = load_settings(settings_path)

    init_logger(settings.log.dir, "image_feature_extractor", logging.DEBUG)

    train_data = pd.read_csv(settings.data.train_path)
    test_data = pd.read_csv(settings.data.test_path)
    species = train_data["species"]
    train_data = train_data.drop("species", axis=1)

    k = np.size(species) * 10
    batch_size = np.size(os.listdir(settings.data.image_path)) * 3
    print(get_feature(settings.data.image_path, test_data['id'], k, batch_size, verbose=False))