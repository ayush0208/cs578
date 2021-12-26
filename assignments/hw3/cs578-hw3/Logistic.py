"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the logistic model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the logistic function but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (logistic) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produces as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the logistic function. 
It is up to your discretion on if you want to use them or add your own methods.
"""
import numpy as np
import Preprocess as Pre
import warnings
warnings.filterwarnings("ignore")

class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the logistic function as instance
        attributes.
        """
        self.vocabulary = None
        self.weights = None
        self.bias = None
        self.n_gram_value = 2
        self.feature_method_extraction = None

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

    def sigmoid(self, z):
	    return(1 / (1 + np.exp(-z)))

    def feature_extraction(self, data, method=None):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training. You need to implement unigram, bigram and trigram.
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

    def logistic_loss(self, predicted_label, true_label):
        """
        Optional helper method to code the loss function.
        """
        return -np.sum(np.dot(true_label, np.log(predicted_label)), np.dot(1-true_label, np.log(1-predicted_label)))
    
    def regularizer(self, lam, method='L2'):
        """
        You need to implement at least L1 and L2 regularizer
        """
        if method == 'L1':
            return lam
        if method == 'L2':
            return lam*self.weights

    def stochastic_gradient_descent(self, data, error, reg_method, lam):
        """
        Optional helper method to compute a gradient update for a single point.
        """
        regularise_value = self.regularizer(lam, reg_method)
        return np.dot(data, error) + regularise_value


    def update_weights(self, learning_rate, gradient):
        """
        Optional helper method to update the weights during stochastic gradient descent.
        """
        new_weights = learning_rate*gradient
        self.weights -= new_weights

    def update_bias(self, learning_rate, error):
        """
        Optional helper method to update the bias during stochastic gradient descent.
        """
        new_bias = np.dot(learning_rate,error)
        self.bias -= new_bias

    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        return np.round(self.sigmoid(np.dot(data_point, self.weights)))

    def train(self, labeled_data, learning_rate=0.001, max_epochs=20, lam=0.001, feature_method='unigram', reg_method='L2'):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.
        
        'learning_rate' and 'max_epochs' are the same as in HW2. 'reg_method' represents the regularier, 
        which can be 'L1' or 'L2' as in the regularizer function. 'lam' is the coefficient of the regularizer term. 
        'feature_method' can be 'unigram', 'bigram' or 'trigram' as in 'feature_extraction' method. Once you find the optimal 
        values combination, update the default values for all these parameters.

        There is no limitation on how you implement the training process.
        """
        self.feature_method_extraction = feature_method
        self.get_vocab(labeled_data, feature_method)
        X, y = Pre.get_features_labels(labeled_data, 'Label')
        X = self.feature_extraction(X, feature_method)
        self.train_after_feature_extraction(X, y, max_epochs, reg_method, lam, learning_rate)
    
    def train_after_feature_extraction(self, X, y, max_epochs, reg_method, lam, learning_rate):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for step in range(0, max_epochs):
            for idx, x_feature in enumerate(X):
                scores = np.dot(x_feature, self.weights) + self.bias
                prediction = self.sigmoid(scores)
                output_error_signal = prediction - y[idx]
                gradient = self.stochastic_gradient_descent(x_feature, output_error_signal, reg_method, lam)
                self.update_weights(learning_rate, gradient)
                self.update_bias(learning_rate, output_error_signal)
        return

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
            predicted_labels.append(self.predict_labels(feature))
        return predicted_labels