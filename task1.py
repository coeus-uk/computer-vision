import utils
import numpy as np
import pandas as pd
from cv2.typing import MatLike

def get_angle_between_lines(img: MatLike, hough_threshold=100, 
                         hough_theta_res=np.deg2rad(1), hough_rho_res=1,
                         canny_gauss_kernel_size=5,canny_gauss_sigma=5,
                         canny_low_threshold=20, canny_high_threshold=40) -> float:

    edges = utils.canny(img, canny_gauss_kernel_size, canny_gauss_sigma,
        canny_low_threshold, canny_high_threshold)

    hough_lines= utils.hough_lines(edges, hough_threshold, hough_theta_res,
        hough_rho_res)

    line_angles = []
    for rho, theta in hough_lines:
        # Handle whether the line is in the positive or negative x
        if (rho < 0):
            theta += np.pi
        line_angles.append(theta)

    if (len(line_angles) < 2):
        print(f"Skipping  - need at least 2 lines")

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
                    hough_lines = utils.hough_lines(image, threshold,
                                                    theta_res, rho_res)
                    
                    line_angles = [theta if rho > 0 else theta + np.pi 
                                   for (rho, theta) in hough_lines]

                    # Can't calcualte angle with one line, set error to inf and move on
                    if (len(line_angles) < 2):
                        results[param_index][image_index] = np.inf
                        continue

                    # Calculate difference of angles and choose the smaller angle
                    angle1 = max(line_angles) - min(line_angles)
                    angle2 = (2 * np.pi) - angle1
                    angle_between_lines = min(angle1, angle2)
                    angle_between_lines = np.round(np.rad2deg(angle_between_lines))

                    error = abs(angle_between_lines - correct_answer)
                    results[param_index][image_index] = error
                    # print(f"Image {image_index} -- theta: {angle_between_lines} -- correct_answer: {correct_answer} -- error: {error}")
        
        total_error = sum(results[param_index])
        print(f"[rho_res, theta_res, threshold] = {params} -- results = {results[param_index]} -- total error: {total_error}")

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