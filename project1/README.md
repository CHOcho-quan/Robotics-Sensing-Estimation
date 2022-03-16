# ECE276 Project 1 README

## Pixel Classification

For pixel classification, I've implemented both the two mentioned methods in class namely Logisitic Regression & Gaussian Discriminant Analysis. To test the classifier, just use

```bash
python3 runClassifier.py --model="GDA" --type="MLE" --feat="RGB" --train=1 --weight=None
```

During testing, please make sure the stored pickle file uses the same feature as your choice of feature and also please input your weight file.

For the meaning of each argument, please see by typing in

```bash
python3 runClassifier.py --help
```

The files in pixel_classification and their implementation are as follow

- GaussianClassifier.py : Implements a general GDA class that can be used in any data classification tasks.
- LogisticRegression.py : Implements a general LR class that can be used in any data classification tasks.
- pixel_classifier.py : Encapsule of the above methods.
- utils.py : Implements basic utilities that is helpful for the classification tasks.

## Bin Detection

For bin detection, I've used the same general GDA and LR class implemented in pixel classification. The two files are almost the same as pixel classification. To test the result simply run

```bash
python3 runBinDetector.py --test_set=data/training
```

It will train the model and simutaneously test it given a specific dataset. If you want to change testing dataset just change the argument test_set to the root folder of your images.

To generate your own training set, please run

```bash
python3 utils.py --from_scratch=0 --start=0 --rgb_set=none --hsv_set=none
```

If you want to select training set not from scratch, please input from_scratch=1 and starting image at start with already given rgb_dataset and hsv_dataset.

The files in bin_detection and their implementation are as follow

- GaussianClassifier.py : Implements a general GDA class that can be used in any data classification tasks.
- LogisticRegression.py : Implements a general LR class that can be used in any data classification tasks.
- bin_detector.py : Encapsule of the above methods and do detection.
- utils.py : Implements basic utilities that is helpful for the classification tasks.