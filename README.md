# Leaf Classification
CZ4041 Machine Learning - Assignmnent. Solution to Kaggle - [Leaf Classification](https://www.kaggle.com/c/leaf-classification) problem.

- [Report](https://www.overleaf.com/15829277brnqrprvmbms#/60282723/)
- [Presentation](https://www.overleaf.com/15402447qhfsrwpgyvng#/58375319/)

## Getting Started

### Requirements
- Python 3.x

### Usage
```bash
pip install -r requirements.txt
python main.py
```

## Contents

### Classifier
This contains various classifier with their respected parameter settings. All the classifiers are tried and tuned to get the best result (lowest log loss). The classifier supported in this repository are:

- K Nearest Neighbors
- Decision Tree
- Naive Bayes
- Gaussian
- Ada Boost
- Gradient Boosting
- SVC
- NuSVC
- Random Forest
- ANN (Artificial Neural Network)
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis


### Utility
There are some utilities can be used for the classifier. The utility modules located at [`utils_leaf_classification`](utils_leaf_classification/).

#### DataLoader
This [module](utils_leaf_classification/data_loader.py) contains the utility to load and preprocess the train and data set.
- **load_train** --- Load train data, separate the id, encode the species and generates all the classes
- **load_test** --- Load test data and separate the id
- **load_from_image** --- Load features from images using SIFT Descriptor

#### DataSelector
This [module](utils_leaf_classification/data_selector.py) contains the utility to add or remove feature from data set.
- **add**|**remove** --- Add/remove a specific feature with specified index
- **add_range**|**remove_range** --- Add/remove a specific feature with all the index in between the given range (`[start, end)`)
- **add_array**|**remove_array** --- Add/remove a specific feature with index specified in the array
- **add_all**|**remove_array** --- Add/remove all index of specified feature (or add/remove all feature if not specified)
- **auto_add_lasso**|**auto_remove_lasso** --- Add/remove feature using LASSO

#### DataReducer
This [module](utils_leaf_classification/data_reducer.py) contains the utility to remove the feature dimension using PCA.
- **pca_data_reduction** --- Reduce feature using PCA dimensionality reduction

#### ModelSelector
This [module](utils_leaf_classification/k_fold.py) contains the utility to select the best classifier by given training data using k-fold cross validation (average result).
- **add_selector** --- Add new data selector with given key
- **add_classifier** --- Add new classifier with given key
- **get_best_model** --- Try all given classifier and choose the best method (using k-fold cross validation)
- **generate_submission** --- Generate submission file based on the specified classifier (or using best classifier if not specified)

#### General Utility
This [module](utils_leaf_classification/utility.py) contains general utility functions.
- **init_logger** --- Initialize logging
- **get_now_str** --- Get now time string (used for naming)
- **ensure_dir** --- Ensure existing directory, to avoid access error
- **load_settings** --- Load settings file as dictionary class

### Settings
The settings file in this project can be used to minimalize code changes. The settings are stored in json file following the format in [default settings](settings.json) file.


### Data
The data directory used to store the data used in the classification process.
- [**Train Data**](data/train.csv)
- [**Test Data**](data/test.csv)
- [**Images**](data/images)
- [**Generated Submission**](data/submission)
- [**Submission Example**](data/submission.csv)


### Logs
All the logging file will be stored in the [log](logs) folder by default.

