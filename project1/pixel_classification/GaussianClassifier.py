# Gaussian Discriminant Analysis
import pickle
import numpy as np

class GDA:
    """
    Gaussian Discriminant Analysis
    
    """

    def __init__(self, weight_file=""):
        """
        Initialization function
        Inputs:
        weight_file - stored weight file for the class

        """
        if weight_file == "":
            self.labels_ = None
            self.label_num_ = None
            self.mu_ = None
            self.sigma_ = None
            self.theta_ = None
        else:
            self.load(weight_file)

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
        self.mu_ = data["mus"]
        self.sigma_ = data["sigmas"]
        self.theta_ = data["thetas"]

    def train(self, X, y, type="MLE"):
        """
        Training function for GDA
        & Store the trained weights
        Inputs:
        X - d x n feature vector with d dim feature
        y - n label of the corresponding feature
        
        """
        d, n = X.shape
        labels = np.unique(y)
        label_num = labels.shape[0]

        # Initialize parameters
        self.labels_ = labels
        self.label_num_ = label_num
        self.mu_ = np.zeros((label_num, d, 1))
        self.sigma_ = np.stack(label_num * [np.zeros((d, d))])
        # If MLE just use non-informative prior
        self.theta_ = np.ones((label_num, 1)) / label_num

        # Start training
        for i in range(label_num):
            label = labels[i]
            curX = np.squeeze(X[:, np.where(y == label)])
            _, cur_num = curX.shape

            # Training for parameters
            self.mu_[i, :, :] = np.sum(curX, axis=1).reshape(d, 1) / cur_num
            self.sigma_[i, :, :] = np.cov(curX)
            if type == "MAP":
                self.theta_[i] = cur_num / n

        # Store values
        data = {
            "labels": self.labels_,
            "label_num": self.label_num_,
            "mus": self.mu_,
            "sigmas": self.sigma_,
            "thetas": self.theta_
        }
        with open("GDA_{0}.pickle".format(type), 'wb') as f:
            pickle.dump(data, f)

    def classify(self, X):
        """
        Classify function
        Inputs:
        X - d x n feature vectors
        Outputs:
        y - 1 x n labels
        
        """
        y = []
        for i in range(self.label_num_):
            prob = 2 * np.log(self.theta_[i]) - \
                np.einsum('ij, ji->i', np.transpose(X - self.mu_[i, :, :]).dot( \
                    np.linalg.inv(self.sigma_[i, :, :])), X - self.mu_[i, :, :]) \
                         - np.log(np.linalg.det(self.sigma_[i, :, :]))
            y.append(prob)
        
        label = self.labels_[np.argmax(np.array(y), axis=0)]
        return label

    def _classify(self, X):
        """
        Do BDR process for GDA
        Inputs:
        X - d x 1 feature vector
        Outputs:
        label - returned label of the feature vector
        
        """
        if self.label_num_ is None:
            print("Untrained model, Error")
            return -666

        result = []
        X = X.reshape(X.shape[0], 1)
        # Calculate BDR for each class
        for i in range(self.label_num_):
            prob = 2 * np.log(self.theta_[i]) \
                    - np.transpose(X - self.mu_[i, :, :]).dot( \
                        np.linalg.inv(self.sigma_[i, :, :])).dot(X - self.mu_[i, :, :]) \
                            - np.log(np.linalg.det(self.sigma_[i, :, :]))
            
            result.append(prob)

        label = self.labels_[np.argmax(np.array(result))]
        return label

