import utils
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from dataclasses import dataclass

@dataclass
class Params:
    hough_threshold: int = 90
    hough_theta_res: float = 1.6689999
    hough_rho_res: float = 0.1
    
    canny_gauss_kernel_size: int = 5
    canny_gauss_sigma: float = 20
    canny_gauss_low_threshold: float = 50
    canny_gauss_high_threshold: float = 150

"""
Empirically tested values that deliver good results. 
"""
PARAMS = Params()

def get_angle_between_lines(img_edges: MatLike, hough_threshold: int, 
                         hough_theta_res: float, hough_rho_res: float) -> float:

    hough_lines= utils.hough_lines(img_edges, hough_threshold, hough_theta_res,
        hough_rho_res)

    line_angles = [theta if rho > 0 else theta + np.pi 
                   for (rho, theta) in hough_lines]

    # Can't calcualte angle with one line, set error to inf and move on
    if (len(line_angles) < 2):
        return np.iinfo("i").max

    # Calculate difference of angles and choose the smaller angle
    angle1 = max(line_angles) - min(line_angles)
    angle2 = (2 * np.pi) - angle1
    angle_between_lines = min(angle1, angle2)
    angle_between_lines = np.round(np.rad2deg(angle_between_lines))

    return angle_between_lines

def get_canny(img, kernel, sigm, low_thresh, up_thresh):

    img_edges = utils.canny(img, kernel, sigm, low_thresh, up_thresh)
    #benchmark_canny = cv2.Canny()

    return img_edges

def try_params(images: list[tuple[np.ndarray, float]], rhos: np.ndarray[float],
               thetas: np.ndarray[float], thresholds: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
    
    all_param_combinations = np.array(np.meshgrid(rhos, thetas, thresholds)).T.reshape(-1,3)
    results = np.zeros((2, len(all_param_combinations), len(images)))
    print(f"Attempting {len(all_param_combinations)} paramter combinations")
    for (param_index, params) in enumerate(all_param_combinations):
        rho_res, theta_res, threshold = params[0], params[1], params[2]
        for (image_index, (image, correct_answer)) in enumerate(images):
                    angle = get_angle_between_lines(images[image_index][0], threshold, theta_res,
                                                     rho_res)

                    results[0][param_index][image_index] = angle
                    results[1][param_index][image_index] = abs(angle - images[image_index][1])
                    # print(f"Image {image_index} -- theta: {angle_between_lines} -- correct_answer: {correct_answer} -- error: {error}")
        
        total_error = np.sum(results[1][param_index])
        print(f"{param_index} [rho_res, theta_res, threshold] = {params} -- results = {results[0][param_index]} -- total error: {total_error}")
        if total_error == 0: print("\n\n\n STOP THE COUNT \n\n\n")
    write_to_csv(results, all_param_combinations)

################################################################


def try_canny_params(image_list: list[(np.ndarray, float)], kernel_size: np.ndarray[int],
               sigma: np.ndarray[float], lower_bound: np.ndarray[int], upper_bound: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
    
    all_param_combinations = np.array(np.meshgrid(kernel_size, sigma, lower_bound, upper_bound)).T.reshape(-1,4)
    
    results = np.zeros((2, len(image_list)))
    print(f"Attempting {len(all_param_combinations)} paramter combinations")
    #pbar = tqdm(total=len(all_param_combinations))
    data_store = []
    for (param_index, params) in enumerate(all_param_combinations):
        results[1] = [0 for i in range(10)]
        for (img_indx, (image, correct_answer)) in enumerate(image_list):
            kernel, sigm, low_thresh, up_thresh = params[0], params[1], params[2], params[3]
            edges = get_canny(image, kernel, sigm, low_thresh, up_thresh)
            
            angle = get_angle_between_lines(edges, PARAMS.hough_threshold,
                                                        PARAMS.hough_theta_res, PARAMS.hough_rho_res)

            error = abs(angle - correct_answer)
            results[0][img_indx] = angle
            results[1][img_indx] = error
            print(f"param - img index:{param_index}/{img_indx} [kernel, sigm, low_thresh, up_thresh] = {params} -- results = {angle} -- errors = {error} -- total error: {np.sum(results[1])}")
        print(" ")
        if np.sum(results[1]) == 0: print("\n\n\n STOP THE COUNT \n\n\n")
        data_store.append([ param_index, img_indx, params[0], 
                                params[1], 
                                params[2], 
                                params[3], 
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
            

    df = pd.DataFrame(data_store, columns= ["param_index", "img_indx","kernel", 
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

        
    return results
    

def write_to_csv(errors: np.ndarray, params: np.ndarray) -> None:
    """
    Construct a CSV consisting of the parameters used for each run, the error
    achieved for each image, and the total error for each run.
    """
    num_params, num_images = errors.shape
    total_errors = np.sum(errors, axis=1)

    columns = ["rho_res", "theta_res", "threshold"]
    columns.extend([f"Image {i + 1} error" for i in range(num_images)])
    columns.append("total_error")

    dataframe = np.zeros((num_params, len(columns)))
    dataframe[:, 0:3] = params
    dataframe[:, 3:3+num_images] = errors
    dataframe[:, 3+num_images] = total_errors

    pd.DataFrame(dataframe, columns=columns).to_csv("results.csv")