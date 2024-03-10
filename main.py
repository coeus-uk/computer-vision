import argparse
import pandas as pd
import cv2
import numpy as np

import task1
import utils


def testTask1(folderName: str) -> int:
    dataset = pd.read_csv(f"{folderName}/list.txt")
    pred_angles = []
    total_error = 0

    print("Precalculate edges for each image...") 
    images = []
    for row in dataset.itertuples():
        filename, correct_answer = row.FileName, row.AngleInDegrees
        img = cv2.imread(f"Task1Dataset/{filename}", cv2.IMREAD_GRAYSCALE)
        edges = utils.canny(img, gauss_kernel_size=25, sigma=2, low_threshold=70, high_threshold=110)
        images.append((edges, correct_answer))

    # Just testing parameters for now
    if True:
        # Adding step to everything because we don't want to include the first 
        #   elem (0) and we want to include last elem. It becomes (0, end].
        rhos = np.arange(0, 2.5, 0.5) + 0.5
        thetas = np.arange(0, 2.5, 0.5) + 0.5
        thresholds = np.arange(0, 200, 40) + 40

        task1.try_params(images, rhos, thetas, thresholds)
    else:
        for row in dataset.itertuples():
            filename, correct_answer = row.FileName, row.AngleInDegrees

            # Read in image, importantly with intensity values 0-255 not 0-1
            img = cv2.imread(f"{folderName}/{filename}", cv2.IMREAD_GRAYSCALE)
            angle = task1.getAngleBetweenLines(img)
            pred_angles.append(angle)

            error = abs(angle - correct_answer)
            total_error += error
            pass_fail_string = "PASS" if error == 0 else "FAIL"
            print(f"{filename} -- theta: {angle} -- correct_answer: {correct_answer} -- error: {error} -- {pass_fail_string}")

    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    return total_error


def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


if __name__ == "__main__":

    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()

    if (args.Task1Dataset == None):
        testTask1("./Task1Dataset")

    if(args.Task1Dataset!=None):
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        testTask1(args.Task1Dataset)
    if(args.IconDataset!=None and args.Task2Dataset!=None):
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask2(args.IconDataset,args.Task2Dataset)
    if(args.IconDataset!=None and args.Task3Dataset!=None):
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask3(args.IconDataset,args.Task3Dataset)
