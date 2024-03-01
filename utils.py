import numpy as np
import cv2

class Utils:
    def canny(self, src: np.ndarray, gauss_kernel_size: int, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
        # Gauss filter
        img = cv2.GaussianBlur(src, (gauss_kernel_size, gauss_kernel_size), sigma)

        # Sobel Filtering
        (theta, magnitude) = self.sobel(img)

        # Non Maximum Suppression
        img = self.non_max_suppression(magnitude, theta, 5)

        # Hystheresis Thresholding
        img = self.hysteresis_thresholding(img,low_threshold, high_threshold)

        return img

        # Sobel filtering
        # NMS
        # Hysterisys thresholding (uses thresholds)
        

    def sobel(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        # theta = np.arctan2(img_x, img_y)
        # magnitude = np.sqrt(np.square(img_x) + np.square(img_y))

        # Apply Sobelx in high output datatype 'float32'
        # and then converting back to 8-bit to prevent overflow

        absx_64 = np.absolute(img_x)
        sobelx_8u1 = absx_64/absx_64.max()*255
        sobelx_8u = np.uint8(sobelx_8u1)

        # Similarly for Sobely
        
        absy_64 = np.absolute(img_y)
        sobely_8u1 = absy_64/absy_64.max()*255
        sobely_8u = np.uint8(sobely_8u1)

        # From gradients calculate the magnitude and changing
        # it to 8-bit (Optional)
        mag = np.hypot(sobelx_8u, sobely_8u)
        mag = mag/mag.max()*255
        mag = np.uint8(mag)

        # Find the direction and change it to degree
        theta = np.arctan2(img_y, img_x)
        angle = np.rad2deg(theta)

        """
        USE TO SEE PLOTS UNCOMMENT ONLY IF YOU WANT TO SEE MIDDLE STEPS                
        """
        # plt.imshow(theta), plt.title('theta'), plt.show()
        # plt.imshow(magnitude), plt.title('magnitudes'), plt.show()
        # plt.imshow(cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)), plt.title('X Derivatives Gradient'), plt.show()
        # plt.imshow(cv2.cvtColor(img_y, cv2.COLOR_GRAY2RGB)), plt.title('Y Derivatives Gradient'), plt.show()

        return (angle, mag)
    
    def non_max_suppression(self, magnitudes: np.ndarray, theta: np.ndarray , window_size: int) -> np.ndarray:
        #theta = np.rad2deg(theta)
        # Find the neighbouring pixels (b,c) in the rounded gradient direction
        # and then apply non-max suppression
        angle = np.rad2deg(theta)
        M, N = magnitudes.shape
        Non_max = np.zeros((M,N), dtype= np.uint8)

        for i in range(1,M-1):
            for j in range(1,N-1):
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

    def hysteresis_thresholding(self, img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
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