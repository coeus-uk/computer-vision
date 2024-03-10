import utils
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from dataclasses import dataclass

@dataclass
class Params:
    hough_threshold: int = 119
    hough_theta_res: float = 1.6689999
    hough_rho_res: float = 0.1
    canny_gauss_kernel_size: int = 25
    canny_gauss_sigma: float = 2
    canny_gauss_low_threshold: float = 70
    canny_gauss_high_threshold: float = 110

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

def try_params(images: list[tuple[MatLike, float]], rhos: np.ndarray[float],
               thetas: np.ndarray[float], thresholds: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
    
    all_param_combinations = np.array(np.meshgrid(rhos, thetas, thresholds)).T.reshape(-1,3)
    results = np.zeros((len(all_param_combinations), len(images)))
    print(f"Attempting {len(all_param_combinations)} paramter combinations")

    for (param_index, params) in enumerate(all_param_combinations):
        rho_res, theta_res, threshold = params[0], params[1], params[2]
        for (image_index, (image, correct_answer)) in enumerate(images):
                    angle = get_angle_between_lines(image, threshold, theta_res,
                                                     rho_res)

                    error = abs(angle - correct_answer)
                    results[param_index][image_index] = error
                    # print(f"Image {image_index} -- theta: {angle_between_lines} -- correct_answer: {correct_answer} -- error: {error}")
        
        total_error = np.sum(results[param_index])
        print(f"{param_index} [rho_res, theta_res, threshold] = {params} -- results = {results[param_index]} -- total error: {total_error}")

    write_to_csv(results, all_param_combinations)

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