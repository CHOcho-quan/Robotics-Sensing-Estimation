# Main functinon to run Classifier
import argparse, pickle
from random import gauss
from wsgiref import validate
from matplotlib.axes import Axes
import numpy as np
import GaussianClassifier as gc
import LogisticRegression as lr
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser(description="Classifier Arguments")
parser.add_argument('--model', type=str, default="GDA", \
    help="Which classifier do you want to use, GDA / LR")
parser.add_argument('--type', type=str, default="MLE", help="MLE / MAP")
parser.add_argument('--feat', type=str, default="RGB", help="Which Feature: RGB / LAB / HSV")
parser.add_argument('--train', type=int, default=1, help="Training / Testing")
parser.add_argument('--weight', type=str, default="", help="Weight file path")

def train_model(classifier, feat, type):
    # Get dataset
    folder = 'data/training'
    X1 = read_pixels(folder+'/red', feat)
    X2 = read_pixels(folder+'/green', feat)
    X3 = read_pixels(folder+'/blue', feat)
    y1, y2, y3 = np.full(X1.shape[0],1), \
        np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
    X, y = np.concatenate((X1,X2,X3)).T, np.concatenate((y1,y2,y3)).T

    # Training
    classifier.train(X, y, type)

def validate_model(classifier, feat):
    # Validate results
    test_fold = 'data/validation'
    Xv1 = read_pixels(test_fold+'/red', feat)
    Xv2 = read_pixels(test_fold+'/green', feat)
    Xv3 = read_pixels(test_fold+'/blue', feat)
    yv1, yv2, yv3 = np.full(Xv1.shape[0],1), \
        np.full(Xv2.shape[0], 2), np.full(Xv3.shape[0], 3)
    Xv, yv = np.concatenate((Xv1, Xv2, Xv3)).T, np.concatenate((yv1, yv2, yv3))

    classifier_result = classifier.classify(Xv)
    yv = yv.reshape(1, Xv.shape[1])
    error = np.sum(np.abs(yv != classifier_result)) / float(Xv.shape[1])
    print("Error count", np.sum(yv != classifier_result), "out of", Xv.shape[1])
    print("Current Classifying Error is", error)

if __name__ == '__main__':
    args = parser.parse_args()
    feat = args.feat

    # Simple Visualization
    folder = 'data/training'
    X1 = read_pixels(folder+'/red', feat)
    X2 = read_pixels(folder+'/green', feat)
    X3 = read_pixels(folder+'/blue', feat)
    y1, y2, y3 = np.full(X1.shape[0],1), \
        np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
    X, y = np.concatenate((X1,X2,X3)).T, np.concatenate((y1,y2,y3)).T

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X1.T[0, :], X1.T[1, :], X1.T[2, :], cmap="Greens")
    ax.scatter(X2.T[0, :], X2.T[1, :], X2.T[2, :], cmap="Greens")
    ax.scatter(X3.T[0, :], X3.T[1, :], X3.T[2, :], cmap="Greens")

    # with open("LR_MLE.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # w = data['omega']
    # x = np.linspace(0, 1, 10)
    # y = np.linspace(0, 1, 10)
    # X1, Y1 = np.meshgrid(x, y)
    # Z1 = (-w[0, 0] * X1 - w[1, 0] * Y1) / w[2, 0]
    # ax.plot_surface(X1, Y1, Z1)

    # X2, Y2 = np.meshgrid(x, y)
    # Z2 = (-w[1, 0] * X2 - w[1, 1] * Y2) / w[1, 2]
    # ax.plot_surface(X2, Y2, Z2)

    # X3, Y3 = np.meshgrid(x, y)
    # Z3 = (-w[2, 0] * X3 - w[2, 1] * Y3) / w[2, 2]
    # ax.plot_surface(X3, Y3, Z3)
    # print(w.shape)
    # plt.show()
    # cv2.imshow("a", y)
    # cv2.waitKey(0)

    if args.model == "GDA":
        if args.train:
            gaussianClassifier = gc.GDA()
            train_model(gaussianClassifier, feat, args.type)
            validate_model(gaussianClassifier, feat)
        else:
            gaussianClassifier = gc.GDA(weight_file=args.weight)
            validate_model(gaussianClassifier, feat)
    elif args.model == "LR":
        if args.train:
            logisticRegressioner = lr.LR()
            train_model(logisticRegressioner, feat, args.type)
            validate_model(logisticRegressioner, feat)
        else:
            logisticRegressioner = lr.LR(weight_file=args.weight)
            validate_model(logisticRegressioner, feat)