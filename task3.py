import cv2 as cv
import numpy as np

from pathlib import Path
from cv2 import DMatch
from cv2.typing import MatLike
from typing import Tuple, List
from dataclasses import dataclass
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

            return cv.imread(str(img_path)), img_path
        else:
            raise StopIteration
        

class BruteForceMatcher:

    def knn_match(
        self, 
        query_descriptors: np.ndarray, 
        train_descriptors: np.ndarray,
        k: int
    ) -> List[Tuple[DMatch, ...]]:
        matches = []
        
        for q_idx, q_desc in enumerate(query_descriptors):
            distances = np.array(
                [self.distance(q_desc, t_desc) for t_desc in train_descriptors]
            )
            # Argsort in ascending order.
            idx_sorted = np.argsort(distances)

            # Find the k closest matches
            k_closest = tuple(
                DMatch(
                    q_idx,
                    idx_sorted[i],
                    None,
                    distances[idx_sorted[i]],
                )
                for i in range(k)
            )
            matches.append(k_closest)

        return matches

    def distance(self, vec1: np.ndarray, vec2: np.ndarray):
        """
        Calculates the Euclidean distance between two vectors.
        """
        return np.linalg.norm(vec1 - vec2)


def axis_aligned_bounding_box(oriented_bounding_box: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Converts an oriented bounding box (OBB) to an axis-aligned bounding box (AABB).

    An AABB is the simplest form of bounding box, where the edges of the box are
    aligned with the axes of the co-ordinate system in which it is defined. That is,
    the sides of the box are parallel to the co-ordinate axes.

    In contrast, an OBB allows for the rotation of the box and is not constrained
    to be aligned with the axes.
    """
    x = oriented_bounding_box[:, 0, 0]
    y = oriented_bounding_box[:, 0, 1]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    return (min_x, min_y, max_x, max_y)


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
        self.matcher = BruteForceMatcher()

    def detect(
        self, 
        test_img: MatLike, 
        draw: bool = True, 
        lowe_ratio_test_threshold: float = 0.7, 
        min_match_count: int = 10
    ):
        detections = []
        kp_test, desc_test = self.sift.detectAndCompute(test_img, None)

        for query_img, img_path in self.query_images:
            # Extract keypoints and generate descriptors using SIFT.
            kp_query, desc_query = self.sift.detectAndCompute(query_img, None)

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
            matches = self.matcher.knn_match(desc_query, desc_test, k=2)
            good_matches = self._perform_lowes_ratio_test(matches, lowe_ratio_test_threshold)

            # We require at least `min_match_count` good matches to be present to find the
            # object.
            if len(good_matches) <= min_match_count:
                print(f"Skipping query image {img_path}. Not enough good matches found - " + \
                      f"{len(good_matches)}/{min_match_count}")
                continue

            try:
                print(f"Query image {img_path} yields enough good matches - " + \
                        f"{len(good_matches)}/{min_match_count}. Finding transform...")
                
                # Extract the locations of matched keypoints in both images.
                source_points = np.float32(
                    [kp_query[m.queryIdx].pt for m in good_matches]
                ).reshape(-1,1,2)
                dest_points = np.float32(
                    [kp_test[m.trainIdx].pt for m in good_matches]
                ).reshape(-1,1,2)

                # `findHomography` needs at least four correct points to find the
                # perspective transformation.
                M, mask = cv.findHomography(source_points, dest_points, cv.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # Define the corner points of the query image.
                h, w, _ = query_img.shape
                query_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                
                # Apply the homography matrix to transform the corner points of the
                # query image to the co-ordinate system of the test image.
                destination = cv.perspectiveTransform(query_corners, M)
                oriented_bounding_box = np.int32(destination)

                polyline_test_img = cv.polylines(
                    test_img.copy(), [oriented_bounding_box], True, 255, 3, cv.LINE_AA)
            except:
                print(f"An exception was raised while handling query image {img_path}")
                continue
            
            if draw:
                matches_img = cv.drawMatches(
                    query_img, kp_query, polyline_test_img, kp_test, good_matches, 
                    None, (0, 255, 0), None, matches_mask, 2)
                
                plt.imshow(matches_img)
                plt.show()

            detections.append((img_path, axis_aligned_bounding_box(oriented_bounding_box)))

        return detections
    
    def _perform_lowes_ratio_test(
        self, 
        matches: List[Tuple[DMatch, ...]],
        lowe_ratio_test_threshold: float
    ) -> List[DMatch]:
        """
        Filters out noisy, ambiguous matches using Lowe's ratio test.

        When writing the report, maybe we should discuss the empirical trade-off between
        false positives & true negatives and how varying this threshold affects them.
        """
        return [ 
            first
            for first, second in matches
            if first.distance / second.distance < lowe_ratio_test_threshold
        ]