import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

import task1
import task2
import utils
import task3

from pathlib import Path
from task3 import ImageDataset


def testTask1(folderName: str) -> int:
    dataset_path = Path(folderName, "list.txt")
    dataset = pd.read_csv(str(dataset_path))

    our_total_error = 0 # error counter
    hough_lines = 0 # inistaite hough_lines
    thresh = 0 # hough threshold
    counter = 0 # initiate conuter for dynamic param assignment
    best_hough = None # keep track of best hough model, initiate to None

    for (i, row) in enumerate(dataset.itertuples()):
        filename, correct_answer = row.FileName, row.AngleInDegrees

        # Read in image, importantly with intensity values 0-255 not 0-1
        img = cv2.imread(str(Path(folderName, filename)), cv2.IMREAD_GRAYSCALE)
        
        #Canny Step 
        edges = utils.canny(img, gauss_kernel_size=5, sigma=35, low_threshold=50, high_threshold=150)

        #Hough Step
        hough_lines = utils.hough_lines(edges, threshold = 90, theta_res= 1.668, rho_res=1)
        best_hough = hough_lines
        
        # Dynamic hyperparameter aassignment
        while len(hough_lines) < 4 or len(hough_lines) > 6:
            if counter == 100: break 
            counter += 1
            if len(hough_lines) > 6: 
                thresh += 1
                hough_lines = utils.hough_lines(edges, threshold = 90 + thresh, theta_res= 1.668, rho_res=1)
                best_hough = best_hough if np.argmin((abs(len(best_hough) - 4), abs(len(hough_lines) - 4), abs(len(best_hough) - 6), abs(len(hough_lines) - 6)))%2 ==0 else hough_lines
            
            else:
                thresh -= 1
                hough_lines = utils.hough_lines(edges, threshold = 90 + thresh, theta_res= 1.668, rho_res=1)
                best_hough = best_hough if np.argmin((abs(len(best_hough) - 4), abs(len(hough_lines) - 4), abs(len(best_hough) - 6), abs(len(hough_lines) - 6)))%2 ==0 else hough_lines

        
        counter = 0
        line_angles = []
        for rho, theta in best_hough:
            # Handle whether the line is in the positive or negative x
            #theta += np.pi
            if (rho < 0):
                #print("error:", filename)
                theta += np.pi

            # theta = theta % (np.pi)
            line_angles.append(theta)

        if(len(line_angles) < 2):
            print(f"Skipping  - need at least 2 lines")

        # Calculate difference of angles and choose the smaller angle
        print(f"max line {np.rad2deg(max(line_angles))} | min line {np.rad2deg(min(line_angles))}")
        
        angle1 = abs(max(line_angles)) - min(line_angles)
        angle2 = (2 * np.pi) - angle1
        print("angle1: " , np.rad2deg(angle1) , "angle2: ", np.rad2deg(angle2))
        
        angle_between_lines = min(angle1, angle2)
        angle_between_lines = np.round(np.rad2deg(angle_between_lines))

        print(f"Our Prediction: {angle_between_lines} -- Correct_answer: {correct_answer} -- Errors: {np.abs(correct_answer-angle_between_lines)}")
        our_total_error += np.abs(correct_answer-angle_between_lines)
        print("\n")

        print(f"Our Total Error: {our_total_error}")
    
    return our_total_error


def testTask2(iconDir: str, testDir: str) -> tuple[float, float, float, float]:
    GAUSS_SIGMA = 5
    GAUSS_KSIZE = (5,5)
    IMG_PYRAMID_SIZE = 1
    NMS_THRESHOLD = 0.005

    images = task2.images_with_annotations(Path(testDir))
    templates = task2.template_pyramid_by_classname(Path(iconDir, "png"))
    templates = [(classname, pyr) for (classname, pyr) in templates]
    template_metrics = {classname:np.array([0, 0, 0]) for (classname, _) in templates}
    results: list[tuple[float, float, float, float]] = []

    for i, (img, annotations) in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, GAUSS_KSIZE, GAUSS_SIGMA)
        img_pyr = task2.gaussian_pyrarmid(gray_img, IMG_PYRAMID_SIZE)

        classnames, scores, pred_boxes = task2.predict_all_templates(img_pyr, templates)
        task2.non_max_suppression(pred_boxes, scores, classnames, NMS_THRESHOLD)
        task2.annotate_predictions(img, classnames, scores, pred_boxes)
        cv2.imwrite(f"{testDir}/results/test_image_{i+1}.png", img)

        predicted_bounds = dict(zip(classnames, pred_boxes))
        metrics = task2.evaluation_metrics(predicted_bounds, annotations, template_metrics)
        results.append(metrics)

    (acc, tpr, fpr, fnr) = [sum(values) / len(values) for values in zip(*results)]
    print(f"Averave accuracy: {acc}")
    print(f"Average true positive rate: {tpr}")
    print(f"Average false positive rate: {fpr}")
    print(f"Average false negative rate: {fnr} ")

    return (acc, tpr, fpr, fnr)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    query_img_dir = Path(iconFolderName, "png")
    test_img_dir = Path(testFolderName, "images")
    test_annotations_dir = Path(testFolderName, "annotations")

    test_images = ImageDataset(test_img_dir, file_ext="png")
    query_images = ImageDataset(query_img_dir, file_ext="png")

    params = {
        "sift_n_features": 0,
        "sift_n_octave_layers": 3, 
        "sift_contrast_threshold": 0.005, 
        "sift_edge_threshold": 11.777774652601527, 
        "sift_sigma": 1.8071337661481155, 
        "ransac_reproj_threshold": 1.0, 
        "ransac_min_datapoints": 4, 
        "ransac_inliers_threshold": 0, 
        "ransac_confidence": 0.9, 
        "lowe_threshold": 0.5,
        "min_match_count": 4
    }

    sift_hps = {
            'nfeatures': params['sift_n_features'],
            'nOctaveLayers': params['sift_n_octave_layers'],
            'contrastThreshold': params['sift_contrast_threshold'],
            'edgeThreshold': params['sift_edge_threshold'],
            'sigma': params['sift_sigma'],
        }

    ransac_hps = {
        'inliers_threshold': params['ransac_inliers_threshold'],
        'min_datapoints': params['ransac_min_datapoints'],
        'reproj_threshold': params['ransac_reproj_threshold'],
        'confidence': params['ransac_confidence']
    }

    lowe_ratio = params['lowe_threshold']
    min_match_count = params['min_match_count']

    return task3.detect_on_dataset(
        test_images, 
        query_images, 
        test_annotations_dir,
        sift_hps,
        ransac_hps,
        lowe_ratio,
        min_match_count,
        verbose=True,
    )


if __name__ == "__main__":

    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()

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
