import numpy as np

class Utils:
    def canny(self, src: np.ndarray, low_threshold: float, high_filter: float) -> np.ndarray:
        # Gauss filter
        return self.gaussian_blur(src, kernel_size=5, sigma=20)

        # Sobel filtering
        # NMS
        # Hysterisys thresholding (uses thresholds)
        

    def convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        'Computes the convolution of an image with a kernel, with clamp-to-edge.'
        outImage = image.copy()
        iDimensions = image.shape

        for iRow in range(iDimensions[0]):
            for iCol in range(iDimensions[1]): 
                accumulator = 0.0
                kDimensions = kernel.shape
                
                for kRow in range(kDimensions[0]):
                    for kCol in range(kDimensions[1]):
                        relativeRow = iRow + kRow - (kDimensions[0] // 2)
                        relativeCol = iCol + kCol - (kDimensions[1] // 2)

                        # Clamping
                        relativeRow = max(0, min(relativeRow, iDimensions[0] - 1))
                        relativeCol = max(0, min(relativeCol, iDimensions[1] - 1))

                        accumulator += image[relativeRow, relativeCol] * kernel[kDimensions[0] - kRow - 1, kDimensions[1] - kCol - 1]
                
                outImage[iRow, iCol] = accumulator

        return outImage

    def gaussian_blur(self, src:np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        kernel = self.gauss_kernel(sigma, kernel_size)
        return self.convolve(src, kernel)

    def gauss_kernel(self, variance: float, size: int):
        if (size % 2 == 0):
            raise Exception("size must be odd")

        gauss = np.zeros([size, size])
        for row in range(gauss.shape[0]):
            x = row - gauss.shape[0] // 2
            for col in range(gauss.shape[1]):
                y = col - gauss.shape[1] // 2
                gauss[row, col] = np.exp(-(x * x +  y * y) / (2.0 * variance)) / (2.0 * np.pi * variance)
        gauss /= gauss.sum()
        
        return gauss