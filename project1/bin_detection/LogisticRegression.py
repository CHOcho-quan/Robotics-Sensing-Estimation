# Logistic Regression
import os, pickle
import numpy as np

class LR:
    """
    Logistic Regression
    
    """

    def __init__(self, weight_file="", regularization=1.0, max_iter=250, learning_rate=1e-3):
        """
        Initialization for parameters
        Inputs:
        weight_file - stored weight file for the class
        
        """
        if weight_file == "":
            self.omega_ = None
            self.labels_ = None
            self.label_num_ = None
        else:
            self.load(weight_file)

        # Training Parameters
        self.lambda_ = regularization
        self.max_iter_ = max_iter
        self.lr_ = learning_rate

    def load(self, weight_file):
        """
        Load pickle file for weights
        Inputs:
        weight_file - stored weight file for the class
        
        """
        with open(weight_file, 'rb') as f:
            data = pickle.load(f)
        self.labels_ = data["labels"]
        self.label_num_ = data["label_num"]
        self.omega_ = data["omega"]

    def _softmax(self, X):
        """
        Softmax function for Logistic Regression
        Inputs:
        X - k x n vector
        Outputs:
        s - k x n probability vector
        
        """
        _, n = X.shape
        return np.exp(X - np.max(X, axis=0)) / \
            np.sum(np.exp(X - np.max(X, axis=0)), axis=0).reshape(1, n)

    def train(self, X, y, type="MLE"):
        """
        Training function for GDA
        & Store the trained weights
        Inputs:
        X - d x n feature vector with d dim feature
        y - 1 x n label of the corresponding feature
        
        """
        d, n = X.shape
        labels = np.unique(y)
        label_num = labels.shape[0]
        onehot_y = np.zeros((y.size, y.max()))
        onehot_y[np.arange(y.size), y - 1] = 1 # n x k onehot label
        onehot_y = onehot_y.T # k x n onehot label

        # Initialize parameters
        self.labels_ = labels
        self.label_num_ = label_num
        self.omega_ = np.random.normal(size=(label_num, d))
        grad = np.zeros_like(self.omega_)

        for i in range(self.max_iter_):
            # Updating omega according to the grad
            grad += (self._softmax(self.omega_.dot(X)) - onehot_y).dot(X.T)
            if type == "MAP":
                grad += 2 * self.lambda_ * self.omega_

            self.omega_ -= self.lr_ * grad
            grad = np.zeros_like(self.omega_)

        # Store values
        data = {
            "labels": self.labels_,
            "label_num": self.label_num_,
            "omega": self.omega_
        }
        with open("LR_{0}_bin.pickle".format(type), 'wb') as f:
            pickle.dump(data, f)

    def classify(self, X):
        """
        Classify function
        Inputs:
        X - d x n feature vectors
        Outputs:
        y - 1 x n labels
        
        """
        if self.omega_ is None:
            print("Untrained model, Error")
            return -666
        pred = self._softmax(self.omega_.dot(X))
        y = np.argmax(pred, axis=0) + 1
        return y
