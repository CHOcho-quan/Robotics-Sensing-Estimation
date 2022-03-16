'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import GaussianClassifier as gc
import LogisticRegression as lr
import numpy as np

class PixelClassifier():
  def __init__(self, type="LR"):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    if type == "GDA":
      self.model_type_ = "GDA"
      self.method_ = "MLE"
      self.model_state_ = False
      self.classifier_ = gc.GDA()
    elif type == "LR":
      self.model_type_ = "LR"
      self.method_ = "MLE"
      self.model_state_ = False
      self.classifier_ = lr.LR()
	
  def train(self, X, y, type="MLE"):
    '''
      Train the classifier
    '''
    self.classifier_.train(X, y, type)
    self.method_ = type
    self.model_state_ = True

  def load_model(self, weight_file):
    '''
      Load model parameters
    '''
    self.classifier_.load(weight_file)
    self.model_state_ = True

  def classify(self, X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
        type: Input type of classifier, namely GDA, LR
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    if self.model_state_:
      y = self.classifier_.classify(X.T)
    else:
      folder_path = os.path.dirname(os.path.abspath(__file__))
      self.load_model("{0}/{1}_{2}.pickle".format(folder_path, self.model_type_, self.method_))
      y = self.classifier_.classify(X.T)
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y.T

