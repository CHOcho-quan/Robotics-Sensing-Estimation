# utils for classifier
import glob
import pickle
import os, cv2
import numpy as np
import argparse
from roipoly import RoiPoly
from matplotlib import pyplot as plt

def add_data(rgb_data, hsv_data, folder, start):
    """
    Add new feature into a existed dataset
    
    """
    bin_pictures = glob.glob(folder + "/*.jpg")
    with open(rgb_data, 'rb') as f:
        rgb_dataset = pickle.load(f)

    with open(hsv_data, 'rb') as f:
        hsv_dataset = pickle.load(f)

    cnt = 0
    cnt_pic = 1
    for img_name in bin_pictures:
        if cnt < start:
            cnt_pic += 1
            cnt += 1
            continue
        print("Current No.", cnt_pic, "picture")
        img = cv2.imread(img_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = hsv_img.astype(np.float64)
        hsv_img[:, :, 0] /= 180.0
        hsv_img[:, :, 1] /= 255.0
        hsv_img[:, :, 2] /= 255.0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        cnt_pic += 1

        y = 1
        # get the image mask
        flag = False
        while y != 3:
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            mask = my_roi.get_mask(img)
            print("Please input the label, 1 for bin blue \
                                        2 for not bin blue \
                                        3 for next picture \
                                        4 for enough")
            y = int(input())

            if y != 3 and y != 4:
                feat = np.mean(img[mask, :], axis=0)
                hsv_feat = np.mean(hsv_img[mask, :], axis=0)
                selected = np.zeros((100, 100, 3))
                selected[:, :, 0] = feat[2] * 255
                selected[:, :, 1] = feat[1] * 255
                selected[:, :, 2] = feat[0] * 255 # BGR
                cv2.imwrite("./compare/{0}/{1}.png".format(y, cnt), selected)
                rgb_dataset[y].append(feat)
                hsv_dataset[y].append(hsv_feat)
            if y == 4:
                flag = True
                break

            cnt += 1
        if flag:
            break

    with open("hsv_dataset_add.pickle", 'wb') as f:
        pickle.dump(hsv_dataset, f)
    with open("rgb_dataset_add.pickle", 'wb') as f:
        pickle.dump(rgb_dataset, f)

def generate_dataset(folder):
    """
    Generate a dataset for GC or LR by ROIpoly
    
    """
    bin_pictures = glob.glob(folder + "/*.jpg")
    rgb_dataset = {1 : [], 2 : []}
    hsv_dataset = {1 : [], 2 : []}

    cnt = 1
    cnt_pic = 1
    for img_name in bin_pictures:
        print("Current No.", cnt_pic, "picture")
        img = cv2.imread(img_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = hsv_img.astype(np.float64)
        hsv_img[:, :, 0] /= 180.0
        hsv_img[:, :, 1] /= 255.0
        hsv_img[:, :, 2] /= 255.0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        cnt_pic += 1

        y = 1
        # get the image mask
        flag = False
        while y != 3:
            fig, ax = plt.subplots()
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            mask = my_roi.get_mask(img)
            print("Please input the label, 1 for bin blue \
                                        2 for not bin blue \
                                        3 for next picture \
                                        4 for enough")
            y = int(input())
            feat = np.mean(img[mask, :], axis=0)
            hsv_feat = np.mean(hsv_img[mask, :], axis=0)
            selected = np.zeros((100, 100, 3))
            selected[:, :, 0] = feat[2] * 255
            selected[:, :, 1] = feat[1] * 255
            selected[:, :, 2] = feat[0] * 255 # BGR
            cv2.imwrite("./compare/{0}/{1}.png".format(y, cnt), selected)

            if y != 3 and y != 4:
                rgb_dataset[y].append(feat)
                hsv_dataset[y].append(hsv_feat)
            if y == 4:
                flag = True
                break

            cnt += 1
        
        if flag:
            break

    with open("hsv_dataset.pickle", 'wb') as f:
        pickle.dump(hsv_dataset, f)
    with open("rgb_dataset.pickle", 'wb') as f:
        pickle.dump(rgb_dataset, f)

def read_pixels(folder, type="RGB", verbose = False):
    """
    Reads 3-D pixel value of the top left corner of each image in folder
        and returns an n x 3 matrix X containing the pixel values 
    
    """
  
    n = len(next(os.walk(folder))[2]) # number of files
    X = np.empty([n, 3])
    i = 0

    if verbose:
        fig, ax = plt.subplots()
        h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
    
    for filename in os.listdir(folder):  
        # read image
        # img = plt.imread(os.path.join(folder,filename), 0)
        img = cv2.imread(os.path.join(folder,filename))
        # convert from BGR (opencv convention) to Other space
        # store pixel rgb value
        if type == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X[i] = img[0, 0].astype(np.float64) / 255.0
        elif type == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            X[i] = img[0, 0].astype(np.float64)
        elif type == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            X[i] = img[0, 0].astype(np.float64)
        i += 1

        # display
        if verbose:
            h.set_data(img)
            ax.set_title(filename)
            fig.canvas.flush_events()
            plt.show()

    return X

parser = argparse.ArgumentParser(description="Dataset Arguments")
parser.add_argument('--from_scratch', type=int, default=1)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--rgb_set', type=str, default="rgb_dataset_add.pickle")
parser.add_argument('--hsv_set', type=str, default="hsv_dataset_add.pickle")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.from_scratch)
    if args.from_scratch:
        generate_dataset('data/training')
    else:
        add_data(args.rgb_set, args.hsv_set, "data/training", args.start)