# Main functinon to run Classifier
import argparse
from random import gauss
from wsgiref import validate
import numpy as np
import bin_detector as bd
import GaussianClassifier as gc
import LogisticRegression as lr
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="Classifier Arguments")
parser.add_argument('--test_set', type=str, default="data/validation")
parser.add_argument('--rgb_model', type=str, default="LR", \
    help="Which RGB classifier do you want to use, GDA / LR")
parser.add_argument('--bin_model', type=str, default="GDA", \
    help="Which Bin classifier do you want to use, GDA / LR")
parser.add_argument('--weight', type=str, default="", help="Weight file path")

if __name__ == '__main__':
    args = parser.parse_args()
    binDetector = bd.BinDetector(args.rgb_model, args.bin_model)

    # Train Bin Model
    # binLR = lr.LR()
    binGDA = gc.GDA()

    # Read Dataset
    with open("./rgb_dataset_add.pickle", 'rb') as f:
        data = pickle.load(f)
    X_1 = np.array(data[1])
    y_1 = np.ones((X_1.shape[0]))
    X_2 = np.array(data[2])
    y_2 = 2 * np.ones((X_2.shape[0]))
    X = np.concatenate([X_1, X_2], axis=0).T
    y = np.concatenate([y_1, y_2], axis=0)

    # with open("./hsv_dataset_add.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # X_1 = np.array(data[1])
    # y_1 = np.ones((X_1.shape[0]))
    # X_2 = np.array(data[2])
    # y_2 = 2 * np.ones((X_2.shape[0]))
    # X = np.concatenate([X_1, X_2], axis=0).T
    # y = np.concatenate([y_1, y_2], axis=0)

    # Simple Visualization
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_1.T[0, :], X_1.T[1, :], X_1.T[2, :], cmap="Greens")
    ax.scatter(X_2.T[0, :], X_2.T[1, :], X_2.T[2, :], cmap="Greens")
    plt.show()

    binGDA.train(X, y)

    # Test Bin Detector
    bin_pictures = glob.glob(args.test_set + "/*.jpg")
    cnt = 1
    for img_name in bin_pictures:
        img = cv2.imread(img_name)
        h, w, _ = img.shape
        befen = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = binDetector.segment_image(img, test_v=True, i=cnt)
        binDetector.get_bounding_boxes(mask, befen, cnt, test_v=True)
        cnt += 1
