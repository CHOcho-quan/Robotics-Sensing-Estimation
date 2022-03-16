'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

from cv2 import fastNlMeansDenoising
import numpy as np
import os, sys, cv2, math, copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import GaussianClassifier as gc
import LogisticRegression as lr
from skimage.measure import label, regionprops

class BinDetector():
	def __init__(self, type1="LR", type2="GDA"):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		if type1 == "GDA":
			self.rgb_model_type_ = "GDA"
			self.rgb_method_ = "MLE"
			self.rgb_model_state_ = False
			self.rgb_model_ = gc.GDA()
		else:
			self.rgb_model_type_ = "LR"
			self.rgb_method_ = "MLE"
			self.rgb_model_state_ = False
			self.rgb_model_ = lr.LR()

		if type2 == "GDA":
			self.bin_model_type_ = "GDA"
			self.bin_method_ = "MLE"
			self.bin_model_state_ = False
			self.bin_model_ = gc.GDA()
		else:
			self.bin_model_type_ = "LR"
			self.bin_method_ = "MLE"
			self.bin_model_state_ = False
			self.bin_model_ = lr.LR()
    		
	def load_model(self, rgb, weight_file):
		'''
			Load Model for classifier
		'''
		if rgb:
			self.rgb_model_.load(weight_file)
			self.rgb_model_state_ = True
		else:
			self.bin_model_.load(weight_file)
			self.bin_model_state_ = True

	def segment_image(self, img, test_v=False, i=1):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv_img = hsv_img.astype(np.float64)
		hsv_img[:, :, 0] /= 180.0
		hsv_img[:, :, 1] /= 255.0
		hsv_img[:, :, 2] /= 255.0

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
		h, w, _ = img.shape
		if not self.rgb_model_state_:
			folder_path = os.path.dirname(os.path.abspath(__file__))
			self.load_model(rgb=True, weight_file="{0}/{1}_{2}.pickle" \
				.format(folder_path, self.rgb_model_type_, self.rgb_method_))
			self.rgb_model_state_ = True
		
		y = self.rgb_model_.classify(img.reshape(-1, 3).T).reshape(h, w)

		if not self.bin_model_state_:
			folder_path = os.path.dirname(os.path.abspath(__file__))
			self.load_model(rgb=False, weight_file="{0}/{1}_{2}_bin.pickle" \
				.format(folder_path, self.bin_model_type_, self.bin_method_))

		# Mask all the non-blue area to white & Run bin classifier
		y[y==3] = 255
		y[y<3] = 0
		img[y!=255, :] = np.array([1.0, 1.0, 1.0])
		hsv_img[y!=255, :] = np.array([1.0, 1.0, 1.0])
		if test_v:
			plt.subplot(121)
			plt.imshow(img)
			plt.subplot(122)
			plt.imshow(y)
			plt.show()
			# plt.savefig("compare/{0}.png".format(i))

		# y_2 = self.bin_model_.classify(hsv_img.reshape(-1, 3).T).reshape(h, w).astype(int)
		y_2 = self.bin_model_.classify(img.reshape(-1, 3).T).reshape(h, w).astype(int)
		y_2[y_2 > 1] = 0
		y_2[y_2 == 1] = 1
		
		if test_v:
			plt.subplot(121)
			plt.imshow(img)
			plt.subplot(122)
			plt.imshow(y_2)
			plt.show()

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y_2

	def get_bounding_boxes(self, y, ori=None, i=1, test_v=False):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach
		h, w = y.shape

		# Morphology Operation
		open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h / 50), int(h / 50)))
		close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h / 100), int(h / 100)))
		dst = cv2.morphologyEx(y.astype(np.uint8), cv2.MORPH_OPEN, open_kernel)
		dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, close_kernel)
		# plt.subplot(121)
		plt.imshow(dst)
		# plt.imshow(ori)
		labels = label(dst, connectivity=1)
		regions = regionprops(labels)
		boxes = []

		for props in regions:
			if props.area <= h / 10.0 * w / 10.0: # too small
				continue
			minr, minc, maxr, maxc = props.bbox
			l = maxc - minc
			w = maxr - minr
			if w / float(l) < 0.8 or w / float(l) > 5:
				continue
			bx = (minc, maxc, maxc, minc, minc)
			by = (minr, minr, maxr, maxr, minr)
			boxes.append([minc, minr, maxc, maxr])
			plt.plot(bx, by, '-b', linewidth=2.5)
			# print(props.area)
		# YOUR CODE BEFORE THIS LINE
		################################################################
		plt.show()
		print(boxes)

		return boxes


