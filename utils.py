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
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    img_x = cv2.filter2D(img, -1, kernel_x)
    img_x = np.absolute(img_x)
    img_x = img_x / img_x.max() * 255
    img_x = img_x.astype(np.uint8)

    img_y = cv2.filter2D(img, -1, kernel_y)
    img_y = np.absolute(img_y)
    img_y = img_y / img_y.max() * 255
    img_y = img_y.astype(np.uint8)

    magnitude = np.hypot(img_x, img_y)
    magnitude = magnitude / magnitude.max() * 255
    magnitude = magnitude.astype(np.uint8)

    theta = np.arctan2(img_y, img_x)

    """
        USE TO SEE PLOTS UNCOMMENT ONLY IF YOU WANT TO SEE MIDDLE STEPS                
        """
    # plt.imshow(theta), plt.title('theta'), plt.show()
    # plt.imshow(magnitude), plt.title('magnitudes'), plt.show()
    # plt.imshow(cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)), plt.title('X Derivatives Gradient'), plt.show()
    # plt.imshow(cv2.cvtColor(img_y, cv2.COLOR_GRAY2RGB)), plt.title('Y Derivatives Gradient'), plt.show()

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
    for (i, row) in enumerate(dirs[1:M-1]):#range(1, M-1):
        # <- this can be an enumeration, directions is only ever accessed at [i,j]
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

# def non_max_suppression(magnitudes: np.ndarray, window_size: int) -> np.ndarray:
#     non_max_suppression = np.empty_like(magnitudes)
#     window_length = window_size // 2
#     padded_magnitude = magnitudes.copy()
#     padded_magnitude = np.pad(
#         padded_magnitude,
#         [(window_length, window_length), (window_length, window_length)],
#           mode='edge')

#     for (i, row) in enumerate(magnitudes):
#         for (j, _) in enumerate(row):
#             right_bound, left_bound = (j + 2 * window_length) + 1, j
#             top_bound, bot_bound = (i + 2 * window_length) + 1, i
#             window = padded_magnitude[bot_bound:top_bound, left_bound:right_bound]
#             non_max_suppression[i, j] = np.amax(window)

#     return non_max_suppression


def hysteresis_thresholding(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    # Set high and low threshold
    M, N = img.shape
    out = np.zeros((M, N), dtype=np.uint8)

    # If edge intensity is greater than 'High' it is a sure-edge
    # below 'low' threshold, it is a sure non-edge
    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    # weak edges
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    # Set same intensity value for all edge pixels
    out[strong_i, strong_j] = 255
    out[zeros_i, zeros_j] = 0
    out[weak_i, weak_j] = 75

    M, N = out.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (out[i, j] == 75):
                if 255 in [out[i+1, j-1], out[i+1, j], out[i+1, j+1], out[i, j-1], out[i, j+1], out[i-1, j-1], out[i-1, j], out[i-1, j+1]]:
                    out[i, j] = 255
                else:
                    out[i, j] = 0

    return out
 
def hough_lines(img: np.ndarray, threshold: int, theta_res: float = np.deg2rad(1), rho_res: float = 1) -> list:
    height, width = img.shape
    max_rho = int(math.hypot(width, height))
    theta_range = np.arange(-np.pi / 2, np.pi / 2, theta_res)
    # print("theta range: ", theta_range)
    rho_range = np.arange(-max_rho, max_rho, rho_res)
    votes = np.zeros((len(rho_range), len(theta_range)), dtype=np.uint16)
    edge_points = np.nonzero(img) # rename to edge_coords or something clearer
    print(np.shape(edge_points))

    # Find points in hough space
    for i in range(len(edge_points[0])):
        y = edge_points[0][i]
        x = edge_points[1][i]
        for t_idx in range(len(theta_range)):
            rho = int(x * np.cos(theta_range[t_idx]) + y * np.sin(theta_range[t_idx]))

            votes[rho + max_rho, t_idx] += 1
    # for point in edge_points:
    #     x, y = point[0], point[1]
    #     for (i, theta) in enumerate(theta_range):
    #         rho  = int(x * np.cos(theta) + y * np.sin(theta))
    #         votes[rho + max_rho, i] += 1
            
    # Find correponding lines for each point in the hough space
    lines = []
    # for (i, row) in enumerate(votes):
    #     for (j, num_votes) in enumerate(row):
    #         if num_votes > threshold:
    #             rho = rho_range[i]
    #             theta = theta_range[j]
    #             lines.append((rho, theta))

    for y in range(votes.shape[0]):
        for x in range(votes.shape[1]):
            if votes[y, x] > threshold:
                rho = rho_range[y]
                theta = theta_range[x]
                lines.append((rho, theta))

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



def hough_line(img: np.ndarray, theta_res=3, rho_res=1):
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


