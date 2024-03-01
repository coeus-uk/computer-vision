import numpy as np
import cv2

def canny(src: np.ndarray, gauss_kernel_size: int, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
    # Gauss filter
    img = cv2.GaussianBlur(src, (gauss_kernel_size, gauss_kernel_size), sigma)

    # Sobel Filtering
    (theta, magnitude) = sobel(img)

    # Non Maximum Suppression
    img = non_max_suppression(magnitude, theta, 5)

    # Hystheresis Thresholding
    img = hysteresis_thresholding(img,low_threshold, high_threshold)

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
    img_y = cv2.filter2D(img, -1, kernel_y)

    theta = np.arctan2(img_x, img_y)
    magnitude = np.sqrt(np.square(img_x) + np.square(img_y))

    """
    USE TO SEE PLOTS UNCOMMENT ONLY IF YOU WANT TO SEE MIDDLE STEPS                
    """
    # plt.imshow(theta), plt.title('theta'), plt.show()
    # plt.imshow(magnitude), plt.title('magnitudes'), plt.show()
    # plt.imshow(cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)), plt.title('X Derivatives Gradient'), plt.show()
    # plt.imshow(cv2.cvtColor(img_y, cv2.COLOR_GRAY2RGB)), plt.title('Y Derivatives Gradient'), plt.show()
    
    return (theta, magnitude)

def non_max_suppression(magnitudes: np.ndarray, angle: np.ndarray , window_size: int) -> np.ndarray:
    # Find the neighbouring pixels (b,c) in the rounded gradient direction
    # and then apply non-max suppression
    angle = np.rad2deg(angle)
    M, N = magnitudes.shape
    Non_max = np.zeros((M,N), dtype= np.uint8)

    for i in range(1, M-1):
        for j in range(1, N-1):
            # Horizontal 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
                b = magnitudes[i, j+1]
                c = magnitudes[i, j-1]
            # Diagonal 45
            elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
                b = magnitudes[i+1, j+1]
                c = magnitudes[i-1, j-1]
            # Vertical 90
            elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
                b = magnitudes[i+1, j]
                c = magnitudes[i-1, j]
            # Diagonal 135
            elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
                b = magnitudes[i+1, j-1]
                c = magnitudes[i-1, j+1]           
                
            # Non-max Suppression
            if (magnitudes[i,j] >= b) and (magnitudes[i,j] >= c):
                Non_max[i,j] = magnitudes[i,j]
            else:
                Non_max[i,j] = 0   

    return Non_max     

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

    #Non_max = img

    M, N = img.shape
    out = np.zeros((M,N), dtype= np.uint8)

    # If edge intensity is greater than 'High' it is a sure-edge
    # below 'low' threshold, it is a sure non-edge
    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    # weak edges
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    # Set same intensity value for all edge pixels
    out[strong_i, strong_j] = 255
    out[zeros_i, zeros_j ] = 0
    out[weak_i, weak_j] = 75

    M, N = out.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (out[i,j] == 75):
                if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
                    out[i, j] = 255
                else:
                    out[i, j] = 0

    return out