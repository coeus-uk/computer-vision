import argparse
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            gauss_kernel_size = np.arange(5, 15, 2)
            gauss_sigma = np.arange(10, 20, 1) 
            gauss_low_threshold = np.arange(30, 60, 10) 
            gauss_high_threshold = np.arange(60, 120, 10)
            #print(img)
            
            edges, combinations = task1.try_canny_params(img, gauss_kernel_size, gauss_sigma, gauss_low_threshold, gauss_high_threshold)
            
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
        
        data_store = []
        results = np.zeros((2, len(images_edges)))
        #print((images_edges[0]))
        for combo_index, combos in enumerate(combinations):
            for (img_index, (image, correct_answer)) in enumerate(images_edges):
                #print(combo_index)
                angle = task1.get_angle_between_lines(image[combo_index], hough_threshold,
                                                    hough_theta_res, hough_rho_res)
                error = abs(angle - correct_answer)
                results[0][img_index] = angle
                results[1][img_index] = error
            print(f"{combo_index} [kernel, sigm, low_thresh, up_thresh] = {combos} -- results = {results[0]} -- errors = {results[1]} -- total error: {np.sum(results[1])}")
            data_store.append([combo_index, 
                                combos[0], 
                                combos[1], 
                                combos[2], 
                                combos[3], 
                                results[0][0], 
                                results[0][1], 
                                results[0][2], 
                                results[0][3], 
                                results[0][4], 
                                results[0][5], 
                                results[0][6], 
                                results[0][7], 
                                results[0][8], 
                                results[0][9],
                                results[1][0], 
                                results[1][1], 
                                results[1][2], 
                                results[1][3], 
                                results[1][4], 
                                results[1][5], 
                                results[1][6], 
                                results[1][7], 
                                results[1][8], 
                                results[1][9],
                                np.sum(results[1])
                                ])
            
            # print("Canny Params= {combos}")
            # print(f"results = {results[0]}")
            # print(f"errors = {results[1]}")
            # print(f"Total error: {np.sum(results[1])}")
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    df = pd.DataFrame(data_store, columns= ["index", 
                                        "kernel", 
                                        "sigma", 
                                        "low_thresh", 
                                        "up_thresh", 
                                        "results1",
                                        "results2",
                                        "results3",
                                        "results4",
                                        "results5",
                                        "results6",
                                        "results7",
                                        "results8",
                                        "results9",
                                        "results10",
                                        "error1",
                                        "error2",
                                        "error3",
                                        "error4",
                                        "error5",
                                        "error6",
                                        "error7",
                                        "error8",
                                        "error9",
                                        "error10",
                                        "cumulative results"])
    df.to_csv("canny_Test.csv")
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
