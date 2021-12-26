"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement the perceptron and the training of the perceptron.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the perceptron model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the perceptron but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (perceptron) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produce as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the perceptron. It is
up to your discretion on if you want to use them or add your own methods.
"""
import numpy as np
import Preprocess as Pre

class Perceptron():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the perceptron as instance attributes.
        """
        self.vocabulary = None
        self.weights = None
        self.bias = None
    
    def get_vocab(self, dataset):
        self.vocabulary = Pre.build_vocab(dataset)

    def feature_extraction(self, data):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in perceptron training.
        """
        df = []
        for i in range(0, data.shape[0]):
            df.append(Pre.get_vector(data.iloc[i]['Text'], self.vocabulary))
        return np.array(df)

    def sgn_function(self, perceptron_input):
        """
        Optional helper method to code the sign function for the perceptron.
        """
        if perceptron_input <=0: return 0
        return 1

    def update_weights(self, update, feature):
        """
        Optional helper method to update the weights of the perceptron.
        """
        new_weights = update * feature
        self.weights += new_weights

    def update_bias(self, update):
        """
        Optional helper method to update the bias of the perceptron.
        """
        self.bias += update

    def predict_labels_cross_Validation(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        prediction = np.dot(data_point, self.weights) + self.bias
        return self.sgn_function(prediction)
    
    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point.
        """
        prediction = np.dot(data_point, self.weights) + self.bias
        return self.sgn_function(prediction)

    def train(self, labeled_data, learning_rate=0.005, max_iter=10):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and max number of iterations. Once you find the
        optimal values of the hyperparameters, update the default values for each keyword argument to reflect those
        values.

        The goal of this function is to train the perceptron on the labeled data. Feel free to code this however you
        want.
        """
        self.get_vocab(labeled_data)
        X, y = Pre.get_features_labels(labeled_data, 'Label')
        X = self.feature_extraction(X)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(max_iter):
            
            for idx, x_feature in enumerate(X):
                output = np.dot(x_feature, self.weights) + self.bias
                y_predicted = self.sgn_function(output)

                update = learning_rate * (y[idx] - y_predicted)
                self.update_weights(update, x_feature)
                self.update_bias(update)
        return

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The first input is the data in the form of a 
        pandas dataframe. 
        
        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers that are either 0 (negative) or 1
        (positive) for each data point.
        
        The rest of the implementation can be fully customized.
        """
        X, y = Pre.get_features_labels(data, 'Label')
        X = self.feature_extraction(X)

        for feat in X:
            predicted_labels.append(self.predict_labels(feat))
        return predicted_labels
