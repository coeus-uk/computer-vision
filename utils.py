import numpy as np
import cv2


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
                b = magnitudes[i+1, j+1]
                c = magnitudes[i-1, j-1]
            # Close to vertical?
            elif (3 * pi_over_8 <= theta < 5 * pi_over_8) or (3 * pi_over_8 - pi <= theta < 5 * pi_over_8 - pi):
                b = magnitudes[i+1, j]
                c = magnitudes[i-1, j]
            # Close to negative diagonal?
            elif (5 * pi_over_8 <= theta < 7 * pi_over_8) or (5 * pi_over_8 - pi <= theta < 7 * pi_over_8 - pi):
                b = magnitudes[i+1, j-1]
                c = magnitudes[i-1, j+1]

            # Non-max Suppression
            if magnitudes[i, j] == max(magnitudes[i, j], b, c):
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
