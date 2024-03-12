import argparse
import pandas as pd
import cv2
import numpy as np

import task1
import utils


def testTask1(folderName: str) -> int:
    canny_testing = True
    dataset = pd.read_csv(f"{folderName}/list.txt")
    total_error = 0

    gauss_kernel_size  = task1.PARAMS.canny_gauss_kernel_size
    gauss_sigma = task1.PARAMS.canny_gauss_sigma
    gauss_low_threshold = task1.PARAMS.canny_gauss_low_threshold
    gauss_high_threshold = task1.PARAMS.canny_gauss_high_threshold


    print("Precalculate edges for each image...") 
    images_edges = []
    for row in dataset.itertuples():
        filename, correct_answer = row.FileName, row.AngleInDegrees
        img = cv2.imread(f"Task1Dataset/{filename}", cv2.IMREAD_GRAYSCALE)

        if canny_testing:
            gauss_kernel_size = np.arange(3, 21, 2)
            gauss_sigma = np.arange(3, 20, 1) 
            gauss_low_threshold = np.arange(20, 60, 5) 
            gauss_high_threshold = np.arange(30, 120, 5)
            
            edges = task1.try_canny_params(img, )
        else:    
            edges = utils.canny(img, gauss_kernel_size, gauss_sigma, 
                                gauss_low_threshold, gauss_high_threshold)
        
        
        images_edges.append((edges, correct_answer))

    # Just testing parameters for now
    if False:
        # Adding step to everything because we don't want to include the first 
        #   elem (0) and we want to include last elem. It becomes (0, end].
        #rhos = np.arange(0, 0.5, 0.05) + 0.05
        rhos = np.array([0.1])
        thetas = np.arange(1.65, 1.68, 0.001) + 0.001
        thresholds = np.arange(118, 121, 0.1) + 0.1

        task1.try_params(images_edges, rhos, thetas, thresholds)
    else:
        hough_threshold = task1.PARAMS.hough_threshold
        hough_theta_res = task1.PARAMS.hough_theta_res
        hough_rho_res = task1.PARAMS.hough_rho_res
        
        results = np.zeros((2, len(images_edges)))
        for (img_index, (image, correct_answer)) in enumerate(images_edges):
            angle = task1.get_angle_between_lines(image, hough_threshold,
                                                  hough_theta_res, hough_rho_res)
            error = abs(angle - correct_answer)
            results[0][img_index] = angle
            results[1][img_index] = error
        
        print(f"results = {results[0]}")
        print(f"errors = {results[1]}")
        print(f"Total error: {np.sum(results[1])}")
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
