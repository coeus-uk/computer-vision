import numpy as np
import cv2

def canny(src: np.ndarray, gauss_kernel_size: int, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Perform canny edge detection on the input image using a series of steps including Gaussian blur, Sobel filtering,
    non-maximum suppression, and hysteresis thresholding.

    Args:
        src (np.ndarray): Input image as a NumPy array.
        gauss_kernel_size (int): Size of the Gaussian kernel for blurring.
        sigma (float): Standard deviation of the Gaussian kernel for blurring.
        low_threshold (float): Lower threshold for hysteresis thresholding.
        high_threshold (float): Higher threshold for hysteresis thresholding.

    Returns:
        np.ndarray: Processed image with detected edges.

    """
    img = src.copy().astype(np.double)
    
    # Apply Gaussian Blur - inbuilt function is ok
    img = cv2.GaussianBlur(img, (gauss_kernel_size, gauss_kernel_size), sigma)
    
    # Sobel Filtering
    dirs, magnitudes = sobel(img)

    # Non Maximum Suppression
    img = non_max_suppression(magnitudes, dirs)

    # Hystheresis Thresholding
    img = hysteresis_thresholding(img, low_threshold, high_threshold)

    return img


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Sobel edge detection to the input image.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - theta (np.ndarray): Gradient direction at each pixel in radians.
            - magnitude_uint8 (np.ndarray): Magnitude of the gradient, scaled to 0-255 and represented as uint8.

    """

    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(img_x**2 + img_y**2)
    max_magnitude = np.max(magnitude) 
    magnitude = (magnitude / max_magnitude) * 255
    #magnitude = magnitude.astype(np.uint8)
    magnitude_uint8 = np.clip(magnitude, 0, 255).astype(np.uint8)
    theta = np.arctan2(img_y, img_x)

    return (theta, magnitude_uint8)


def non_max_suppression(magnitudes: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    '''
    Find the largest magnitude in each direction within a 3x3 sliding window.
    Iterate over the image with a 3x3 sliding window, breaks the window into 
    8 directions and find the maximum magnitude along each of them.

    E.g. here the angle of pixel '@' is close to the diagonal passing through
    ##b  c & b so it is compared to those neighbouring pixels.
    #@#
    c##

    #b# here the angle is close to the vertical, meaning it is compared against
    #@# the neighbours above and below.
    #c#
    '''

    if (magnitudes.shape != dirs.shape):
        raise ValueError("Must have same nmber of magnitudes as directions")

    M, N = dirs.shape
    result = np.zeros_like(dirs)

    for i in range(1, M-1):
        for j in range(1, N-1):
            theta = dirs[i, j]
            angle_bin = round(theta / (np.pi / 4)) % 4

            if angle_bin == 0:  # Horizontal
                neighbors = [magnitudes[i, j-1], magnitudes[i, j+1]]
            elif angle_bin == 1:  # Positive diagonal
                neighbors = [magnitudes[i-1, j-1], magnitudes[i+1, j+1]]
            elif angle_bin == 2:  # Vertical
                neighbors = [magnitudes[i-1, j], magnitudes[i+1, j]]
            else:  # Negative diagonal
                neighbors = [magnitudes[i-1, j+1], magnitudes[i+1, j-1]]

            if magnitudes[i, j] >= max(neighbors):
                result[i, j] = magnitudes[i, j]

    return result

def hysteresis_thresholding(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    '''
    Filter edges such that all pixels with a value below `low_threshold` become 0
    and all edges with a value above `high_threshold` become the max value, 255.

    Pixels with an intensity value between the two thresholds are set to 255 if 
    another maximum exists in the neighbourhood and 0 if no maximum exists.

    This can be thought of enforcing that weak edges must be connected to a 
    strong edge. I.e. a neighbourhood of weak edges will get reduced to 0, 
    but a neighbourhood of weakedges next to a strong edge will survive.
    '''
    M, N = img.shape
    out = np.zeros((M, N), dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    out[strong_i, strong_j] = 255
    out[zeros_i, zeros_j] = 0
    out[weak_i, weak_j] = 75 #<- if we enumerated the indices we have we wouldn't need to set this to 75 first. Setting to an arbitrary number looks weird.

    neighbourhood = lambda i, j: [out[i+1, j-1], out[i+1, j], out[i+1, j+1],
                                  out[i, j-1],                out[i, j+1], 
                                  out[i-1, j-1], out[i-1, j], out[i-1, j+1]]

    for i in range(1, M-1): # <- we already have the indices of the weak edges, why not enumerate them rather than all pixels?
        for j in range(1, N-1):
            if (out[i, j] == 75): # <- should this be a copy? we are mutating the same array our condition is based off.
                if 255 in neighbourhood(i,j):
                    out[i, j] = 255
                else:
                    out[i, j] = 0

    return out

def hough_lines(img: np.ndarray, threshold:int=80, theta_res: float=1.667, rho_res: float=1) -> list[tuple]:
    """
    Detect lines in the input image using the Hough transform.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        threshold (int): Threshold for line detection. Default is 80.
        theta_res (float): Resolution of theta values in degrees. Default is 1.667.
        rho_res (float): Resolution of rho values. Default is 1.

    Returns:
        list[tuple]: List of detected lines represented as (rho, theta) tuples.
    """

    accumulator, theta_range, rho_range = hough_transform(img, theta_res, rho_res)
    return get_lines(accumulator, theta_range, rho_range, threshold)

def hough_transform(img: np.ndarray, theta_res: float=3, rho_res: float=1) -> tuple[np.ndarray[np.uint8], np.ndarray[float], np.ndarray[float]]:
    """
    Apply Hough transform on the input image to detect lines.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        theta_res (float): Resolution of theta values in degrees. Default is 3.
        rho_res (float): Resolution of rho values. Default is 1.

    Returns:
        tuple: A tuple containing:
            - accumulator (np.ndarray[np.uint8]): Accumulator array.
            - theta_range (np.ndarray[float]): Range of theta values.
            - rho_range (np.ndarray[float]): Range of rho values.
    """
    height, width = img.shape
    max_rho = np.hypot(height, width).astype(int) 
    theta_range = np.deg2rad(np.arange(-90, 90, theta_res))
    rho_range = np.arange(-max_rho, max_rho, rho_res)
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int64)

    # For each point in cartesian coordiantes (x,y) lying on an edge, 
    #  calculate a range of lines in polar coordinates (r, theta) that pass
    #  through the point. 
    #  Recall that rho = x * cos(theta) + y * sin(theta).
    y_vals, x_vals = np.where(img > 0)

    cos_vals = np.cos(theta_range)
    sin_vals = np.sin(theta_range)

    y_sin_theta = np.array([y * sin_vals for y in y_vals])
    x_cos_theta = np.array([x * cos_vals for x in x_vals])

    # Row index -> index of rho in rho_range
    # Col index -> index of theta in theta range
    rho_vals = np.round(x_cos_theta + y_sin_theta).astype(int)

    # Map values from [-max_rho, max_rho] to [0, 2*max_rho]
    rho_vals += max_rho

    # Map values of rho to index of that value in rho_range
    rho_indexes = (rho_vals // rho_res).astype(np.int32)

    # For each possible angle, determine which values of rho are used to 
    #  construct lines through each point, as well as how many time it is used.
    #  Add this to the total number of votes for this pair of (rho, theta)
    for t_idx in range(len(theta_range)):
        unique_rho_indexes, counts = np.unique(rho_indexes[:, t_idx], return_counts=True)
        accumulator[unique_rho_indexes, t_idx] += counts

    return accumulator, theta_range, rho_range

# This function can definately be combined with hough_line, it is only ever called as a combination with hough_line and vice versa.
# This would simplify the API.
def get_lines(accumulator, theta_range, rho_range, threshold):
    """
    Extract lines from the accumulator array based on the given threshold.

    Args:
        accumulator (np.ndarray): Accumulator array obtained from Hough transform.
        theta_range (np.ndarray): Range of theta values.
        rho_range (np.ndarray): Range of rho values.
        threshold (int): Threshold for line detection.

    Returns:
        list[tuple]: List of detected lines represented as (rho, theta) tuples.
    """
    y_idxs, x_idxs = np.where(accumulator > threshold)
    rho_vals = rho_range[y_idxs]
    theta_vals = theta_range[x_idxs]
    lines = list(zip(rho_vals, theta_vals))
    return lines



