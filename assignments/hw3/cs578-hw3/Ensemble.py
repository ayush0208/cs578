"""
You only need to implement bagging.
"""
import numpy as np
import Preprocess as Pre
import Logistic as LR
import random 
from multiprocessing import Pool
from statistics import mode 
import warnings
warnings.filterwarnings("ignore")

class Ensemble():
    def __init__(self):
        """
        You may initialize the parameters that you want and remove the 'return'
        """
        self.vocabulary = None
        self.weights = None
        self.bias = None
        self.n_gram_value = 1
        self.clf_list = []
        self.num_clf = None
        self.feature_method_extraction = 'unigram'
    
    def get_vocab(self, dataset, method):
        if method == 'unigram':
            self.vocabulary = Pre.build_vocab_n_gram(dataset, 1)
            self.n_gram_value = 1
        if method == 'bigram':
            self.vocabulary = Pre.build_vocab_n_gram(dataset, 2)
            self.n_gram_value = 2
        if method == 'trigram':
            self.vocabulary = Pre.build_vocab_n_gram(dataset, 3)
            self.n_gram_value = 3

    def feature_extraction(self, data, method='unigram'):
        """
        Use the same method as in Logistic.py
        """
        df = []
        if method == 'unigram':
            for i in range(0, data.shape[0]):
                df.append(Pre.get_vector_n_gram(data.iloc[i]['Text'], self.vocabulary, self.n_gram_value))
            return np.array(df)

        if method == 'bigram':
            for i in range(0, data.shape[0]):
                df.append(Pre.get_vector_n_gram(data.iloc[i]['Text'], self.vocabulary, self.n_gram_value))
            return np.array(df)
        
        if method == 'trigram':
            for i in range(0, data.shape[0]):
                df.append(Pre.get_vector_n_gram(data.iloc[i]['Text'], self.vocabulary, self.n_gram_value))
            return np.array(df)

    def train(self, labeled_data, num_clf=31):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.

        There is no limitation on how you implement the training process.
        """
        self.num_clf = num_clf
        self.get_vocab(labeled_data, self.feature_method_extraction)
        X, y = Pre.get_features_labels(labeled_data, 'Label')
        X = self.feature_extraction(X, self.feature_method_extraction)
        self.train_multiple_LR(X, y)
    
    def train_multiple_LR(self, X, y):
        np.random.seed(0)
        for k in range(self.num_clf):
            indices_taken = np.random.choice(np.arange(len(X)), size = len(X), replace = True)
            log = LR.Logistic()
            log.train_after_feature_extraction(X[indices_taken], y[indices_taken], max_epochs=100, reg_method='L2', lam=0.001, learning_rate=0.001)
            self.clf_list.append(log)

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        X, y = Pre.get_features_labels(data, 'Label')
        X = self.feature_extraction(X, self.feature_method_extraction)

        for feature in X:
            predicted_labels.append(self.predict_instance(feature))
        return predicted_labels
    
    def predict_instance(self, instance):
        bagging_list = []
        for classifier in self.clf_list:
            bagging_list.append(classifier.predict_labels(instance))
        return mode(bagging_list)

    

