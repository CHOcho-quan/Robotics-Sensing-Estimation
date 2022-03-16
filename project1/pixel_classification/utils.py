# utils for classifier
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2

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
            float_img = img.astype(np.float64)
            X[i] = float_img[0, 0] / 255.0
        elif type == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            float_img = img.astype(np.float64)
            float_img[0, 0, 0] /= 180.0
            float_img[0, 0, 1] /= 255.0
            float_img[0, 0, 2] /= 255.0
            X[i] = float_img[0, 0]
        i += 1

        # display
        if verbose:
            h.set_data(img)
            ax.set_title(filename)
            fig.canvas.flush_events()
            plt.show()

    return X