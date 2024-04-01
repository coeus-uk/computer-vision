import cv2 as cv
import numpy as np

from pathlib import Path
from cv2.typing import MatLike
from typing import Tuple
from typing_extensions import Self
from matplotlib import pyplot as plt


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
    
    def __next__(self) -> Tuple[MatLike, Path]:
        """
        Returns the next image in the directory as well as its path.
        """
        if self.index < len(self.img_paths):
            img_path = self.img_paths[self.index]
            self.index += 1

            return cv.imread(str(img_path), cv.IMREAD_UNCHANGED), img_path
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

    def detect(self, test_img: MatLike, lowe_ratio_test_threshold: float = 0.7, min_match_count: int = 10):
        for query_img, img_path in self.query_images:
            # Extract keypoints and generate descriptors using SIFT.
            kp_query, desc_query = self.sift.detectAndCompute(query_img, None)
            kp_test, desc_test = self.sift.detectAndCompute(test_img, None)

            # For each descriptor in `desc_query`, find the `k` closest descriptors in 
            # `desc_test`. We choose `k=2` for use in applying Lowe's ratio test, which
            # is a method to filter out poor matches (outliers?). Lowe's ratio test is
            # a heuristic to select good matches between two sets of features. By asking
            # `knnMatch` to find the 2 nearest neighbours, we can apply this test, which
            # compares the distance of the closest neighbour to the second closest
            # neighbour. The test asserts that if the ratio of the closest distance to
            # the second closest distance is below acertain threshold (e.g., 0.7), then
            # the match is considered good. The rational is that a good match is
            # significantly closer than the second best match. This helps to filter out
            # many false matches (FPs) where the difference between the best and second-
            # best is not significant.
            matches = self.matcher.knnMatch(desc_query, desc_test, k=2)

            # Lowe's ratio test. You can think of this as filtering out noisy, ambiguous
            # matches. When writing the report, maybe we should discuss the empirical 
            # trade-off between false positives & true negatives by varying this threshold.
            good_matches = []
            for first, second in matches:
                if first.distance / second.distance < lowe_ratio_test_threshold:
                    good_matches.append(first)

            if len(good_matches) > min_match_count:
                try:
                    print(f"Query image {img_path} yields enough good matches. Finding transform...")
                    source_points = np.array([kp_query[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1,1,2)
                    dest_points = np.array([kp_test[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1,1,2)

                    # We need to implement `findHomography`, `RANSAC`.
                    M, mask = cv.findHomography(source_points, dest_points, cv.RANSAC, 5.0)
                    matches_mask = mask.ravel().tolist()

                    h, w, c = query_img.shape
                    points = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype=np.float32).reshape(-1,1,2)
                    destination = cv.perspectiveTransform(points, M)

                    polyline_test_img = cv.polylines(test_img.copy(), [np.int32(destination)], True, 255, 3, cv.LINE_AA)
                except:
                    # matches_mask = None
                    continue
            else:
                print(f"Skipping query image {img_path}. Not enough good matches found - {len(good_matches)}/{min_match_count}")
                # matches_mask = None
                continue

            draw_params = {
                "matchColor": (0, 255, 0), # Draw matches in green.
                "singlePointColor": None,
                "matchesMask": matches_mask, # Only draw inliers.
                "flags": 2,
            }

            matches_img = cv.drawMatches(query_img, kp_query, polyline_test_img, kp_test, good_matches, None, **draw_params)
            plt.imshow(matches_img)
            plt.show()