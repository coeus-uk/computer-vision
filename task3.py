import cv2 as cv
import numpy as np

from pathlib import Path
from cv2.typing import MatLike
from typing_extensions import Self


class ImageDataset:
    """
    Dataset of images with lazy loading.

    Iterates over images in a given directory, loading them one-by-one as needed.
    """
    def __init__(self, img_dir_path: Path, file_ext: str):
        self.img_dir_path = img_dir_path
        self.img_paths = list(img_dir_path.glob(f'*.{file_ext}'))
        self.index = 0

    def __iter__(self) -> Self:
        """
        Returns itself as an iterator object.
        """
        # Reset the counter for every new iteration.
        self.index = 0
        return self
    
    def __next__(self) -> MatLike:
        """
        Returns the next image in the directory.
        """
        if self.index < len(self.img_paths):
            img_path = self.img_paths[self.index]
            self.index += 1

            return cv.imread(str(img_path), cv.IMREAD_UNCHANGED)
        else:
            raise StopIteration


class ObjectDetector:
    """
    Object detector using SIFT keypoint localisation and descriptors.
    """
    def __init__(
            self, 
            query_images: ImageDataset, 
            sift_hyperparams: dict,
            flann_index_params: dict,
            flann_search_params: dict,
        ):
        self.query_images = query_images

        self.sift = cv.SIFT.create(**sift_hyperparams)
        self.matcher = cv.FlannBasedMatcher(flann_index_params, flann_search_params)

    def detect(self, test_image: MatLike):
        pass