import numpy as np
import cv2
import math


def canny(src: np.ndarray, gauss_kernel_size: int, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
    img = src.copy().astype(np.double)
    img = cv2.GaussianBlur(img, (gauss_kernel_size, gauss_kernel_size), sigma)
    (dirs, magnitudes) = sobel(img)
    img = non_max_suppression(magnitudes, dirs)

    # Hystheresis Thresholding
    img = hysteresis_thresholding(img, low_threshold, high_threshold)

    return img


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # kernel_x = np.array([
    #     [-1, 0, 1],
    #     [-2, 0, 2],
    #     [-1, 0, 1]
    # ])
    # kernel_y = np.array([
    #     [-1, -2, -1],
    #     [0, 0, 0],
    #     [1, 2, 1]
    # ])

    # img_x = cv2.filter2D(img, -1, kernel_x)
    # img_x = np.absolute(img_x)
    # img_x = img_x / img_x.max() * 255
    # img_x = img_x.astype(np.uint8)

    # img_y = cv2.filter2D(img, -1, kernel_y)
    # img_y = np.absolute(img_y)
    # img_y = img_y / img_y.max() * 255
    # img_y = img_y.astype(np.uint8)

    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    magnitude = np.hypot(img_x, img_y)
    magnitude = magnitude / magnitude.max() * 255
    magnitude = magnitude.astype(np.uint8)

    theta = np.arctan2(img_y, img_x)

    return (theta, magnitude)


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
    # dirs = np.rad2deg(dirs.copy()) # <- personally I'd find fractions of pi easier to read than decimals of degrees

    # convert range of angles from range [-2pi, 2pi] to [0, 2pi]
    # dirs = (dirs + (2 * np.pi)) % (2 * np.pi)

    if (magnitudes.shape != dirs.shape):
        raise ValueError("Must have same nmber of magnitudes as directions")

    M, N = dirs.shape
    result = np.empty_like(dirs)
    pi_over_8 = np.pi / 8
    pi = np.pi

    # Not padding for the moment but that's ok, can always try that later
    for (i, row) in enumerate(dirs[1:M-1]):
        for (j, theta) in enumerate(row[1:N-1]):
            # Close to horizontal?
            if (-pi_over_8 <= theta < pi_over_8) or (pi - pi_over_8 <= theta <= pi) or (-pi <= theta < pi_over_8 - pi):
                b = magnitudes[i, j+1]
                c = magnitudes[i, j-1]
            # Close to positive diagonal?
            elif (pi_over_8 <= theta < 3 * pi_over_8) or (pi_over_8 - pi <= theta < 3 * pi_over_8 - pi):
                b = magnitudes[i+1, j-1]
                c = magnitudes[i-1, j+1]
            # Close to vertical?
            elif (3 * pi_over_8 <= theta < 5 * pi_over_8) or (3 * pi_over_8 - pi <= theta < 5 * pi_over_8 - pi):
                b = magnitudes[i+1, j]
                c = magnitudes[i-1, j]
            # Close to negative diagonal?
            elif (5 * pi_over_8 <= theta < 7 * pi_over_8) or (5 * pi_over_8 - pi <= theta < 7 * pi_over_8 - pi):
                b = magnitudes[i+1, j+1]
                c = magnitudes[i-1, j-1]

            # TODO: investigate best weight for this
            weight = np.abs(np.tan(theta))
            interpolated_mag = int(b) * weight + int(c) * (1 - weight)

            # Non-max Suppression
            if magnitudes[i, j] == max(magnitudes[i, j], interpolated_mag):
                result[i, j] = magnitudes[i, j]
            else:
                result[i, j] = 0

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

def hough_lines(img: np.ndarray, threshold:int=20, theta_res: float=3, rho_res: float=1) -> list[tuple]:
    accumulator, theta_range, rho_range = hough_line(img, theta_res, rho_res)
    return get_lines(accumulator, theta_range, rho_range, threshold)

def hough_line(img: np.ndarray, theta_res: float=3, rho_res: float=1) -> tuple[np.ndarray[np.uint8], np.ndarray[float], np.ndarray[float]]:
    height, width = img.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    theta_range = np.deg2rad(np.arange(-90, 90, theta_res))
    rho_range = np.arange(-max_rho, max_rho, rho_res)
    num_thetas = len(theta_range)
    accumulator = np.zeros((2 * max_rho, num_thetas), dtype=np.uint8)

    edge_points = np.argwhere(img)

    x_vals = edge_points[:, 1]
    y_vals = edge_points[:, 0]

    cos_vals = np.cos(theta_range)
    sin_vals = np.sin(theta_range)

    rho_vals = np.round(x_vals[:, None] * cos_vals + y_vals[:, None] * sin_vals).astype(np.int)
    rho_vals += max_rho

    for t_idx in range(num_thetas):
        rho, count = np.unique(rho_vals[:, t_idx], return_counts=True)
        accumulator[rho, t_idx] = count

    return accumulator, theta_range, rho_range

def get_lines(accumulator, theta_range, rho_range, threshold):
    y_idxs, x_idxs = np.where(accumulator > threshold)
    rho_vals = rho_range[y_idxs]
    theta_vals = theta_range[x_idxs]
    lines = list(zip(rho_vals, theta_vals))
    return lines

def draw_lines(img, lines):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


