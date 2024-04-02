import cv2 as cv
import numpy as np

from pathlib import Path
from cv2 import DMatch, SIFT
from cv2.typing import MatLike
from dataclasses import dataclass
from typing_extensions import Self
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Tuple, List, Callable


def squared_error_loss(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Calculates the point-wise squared difference of two vectors.
    """
    return (truth - prediction) ** 2

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two vectors.
    """
    return np.linalg.norm(vec1 - vec2)


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

    def __init__(self, distance_metric: Callable[[np.ndarray, np.ndarray], float]):
        self.distance = distance_metric

    def knn_match(self, src_vecs: np.ndarray, dst_vecs: np.ndarray, k: int) -> List[Tuple[DMatch, ...]]:
        matches = []
        
        # Compute the distance between every pairing. Keep only the closest.
        for s_idx, s_vec in enumerate(src_vecs):
            distances = np.array(
                [self.distance(s_vec, d_vec) for d_vec in dst_vecs]
            )
            # Argsort in ascending order.
            idx_sorted = np.argsort(distances)

            # Find the k closest matches.
            k_closest = tuple(
                DMatch(
                    s_idx,
                    idx_sorted[i],
                    None,
                    distances[idx_sorted[i]],
                )
                for i in range(k)
            )
            matches.append(k_closest)

        return matches


class IModel(ABC):

    @abstractmethod
    def fit(self, src, dst):
        pass

    @abstractmethod
    def transform(self, src):
        pass

    @abstractmethod
    def reset(self):
        pass


class RANSAC(IModel):

    def __init__(
            self, 
            min_datapoints: int, 
            max_iterations: int, 
            threshold: float, 
            model_cls: IModel, 
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
            model_hyperparams: dict,
            ):
        self.min_datapoints = min_datapoints
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.model_cls = model_cls
        self.loss = loss
        self.model_hyperparams = model_hyperparams
        self.best_model: IModel = None
        self.max_inliers = 0

    def fit(self, src: np.ndarray, dst: np.ndarray):
        assert(src.shape == dst.shape)
        num_points = src.shape[1]

        for iteration in range(self.max_iterations):
            permuted_idxs = np.random.permutation(num_points)

            sample_idxs = permuted_idxs[:self.min_datapoints]
            src_samples = src[:, sample_idxs]
            dst_samples = dst[:, sample_idxs]

            model = self.model_cls(**self.model_hyperparams)
            model = model.fit(src_samples, dst_samples)

            remaining_idxs = permuted_idxs[self.min_datapoints:]
            src_non_samples = src[:, remaining_idxs]
            dst_non_samples = dst[:, remaining_idxs]

            losses = self.loss(dst_non_samples, model.transform(src_non_samples))
            within_threshold = losses < self.threshold

            inlier_idxs = remaining_idxs[within_threshold]

            if len(inlier_idxs) > self.max_inliers:
                self.max_inliers = len(inlier_idxs)

                inlier_idxs = np.hstack([sample_idxs, inlier_idxs])
                src_inliers = src[:, inlier_idxs]
                dst_inliers = dst[:, inlier_idxs]

                better_model = self.model_cls(**self.model_hyperparams)
                better_model.fit(src_inliers, dst_inliers)

                self.best_model = better_model

    def transform(self, src: np.ndarray) -> np.ndarray:
        return self.best_model.transform(src)
    
    def reset(self):
        self.best_model = None
        self.max_inliers = 0


class DirectLinearTransformer(IModel):

    def __init__(self):
        self.reset()

    def fit(self, src: np.ndarray, dst: np.ndarray) -> Self:
        """
        Estimates a 2D homographic transformation using a generalisation of the four-
        point algorithm.

        Assumes that `src` and `dst` are arrays of homogeneous co-ordinates that have 
        shape `(3, n)`, where `n` is the number of points.
        """
        assert(src.shape == dst.shape)
        num_points = src.shape[1]

        # Stack the equations into a homogeneous linear system.
        A = np.zeros((2*num_points, 9))
        for i in range(num_points):
            A[2*i, 0:3] = src[:, i]
            A[2*i, 6:9] = -dst[0, i] * src[:, i]
            A[2*i+1, 3:6] = src[:, i]
            A[2*i+1, 6:9] = -dst[1, i] * src[:, i]

        # Solve the homogeneous linear system using SVD
        U, D, Vt = np.linalg.svd(A)
        H = Vt[-1, :].reshape(3, 3)

        # Normalise the solution to ensure H[2,2] == 1
        self.homography = H / H[2, 2]

        return self

    def transform(self, src: np.ndarray) -> np.ndarray:
        return self.homography @ src
    
    def reset(self):
        self.homography = np.zeros((3,3))


@dataclass
class AlignedBoundingBox:
    """
    Axis-aligned bounding box (AABB).

    An AABB is the simplest form of bounding box, where the edges of the box are
    aligned with the axes of the co-ordinate system in which it is defined. That is,
    the sides of the box are parallel to the co-ordinate axes.
    """

    points: np.ndarray


class BoundingBox:
    """
    Oriented bounding box (OBB).

    An OBB is not constrained to be aligned with the axes. That is, its edges are not
    necessarily parallel to the co-ordinate axes.
    """

    def __init__(self, oriented_points: np.ndarray):
        self.points = oriented_points

    def align_with_axis(self) -> AlignedBoundingBox:
        """
        Converts an oriented bounding box (OBB) to an axis-aligned bounding box (AABB).

        An AABB is the simplest form of bounding box, where the edges of the box are
        aligned with the axes of the co-ordinate system in which it is defined. That is,
        the sides of the box are parallel to the co-ordinate axes.

        In contrast, an OBB allows for the rotation of the box and is not constrained
        to be aligned with the axes.
        """
        x = self.points[:, 0, 0]
        y = self.points[:, 0, 1]

        axis_aligned_points = np.array([min(x), min(y), max(x), max(y)])
        return AlignedBoundingBox(axis_aligned_points)


@dataclass
class Detection:

    img_path: Path
    bounding_box: BoundingBox


class ObjectDetector:
    """
    Object detector using SIFT keypoint localisation and descriptors.
    """
    def __init__(self, query_images: ImageDataset, sift_hyperparams: dict):
        self.query_images = query_images

        self.sift = SIFT.create(**sift_hyperparams)
        self.matcher = BruteForceMatcher(euclidean_distance)
        self.ransac = RANSAC(4, 14, 5, DirectLinearTransformer, squared_error_loss, {})

    def detect(
            self, 
            train_img: MatLike, 
            draw: bool = True, 
            lowe_ratio_test_threshold: float = 0.7, 
            min_match_count: int = 10
            ) -> List[Detection]:
        
        detections = []
        kp_train, desc_train = self.sift.detectAndCompute(train_img, None)

        for query_img, img_path in self.query_images:
            # Extract keypoints and generate descriptors using SIFT.
            kp_query, desc_query = self.sift.detectAndCompute(query_img, None)

            # For each descriptor in `desc_query`, find the `k` closest descriptors in 
            # `desc_train`. We choose `k=2` for use in applying Lowe's ratio test, which
            # is a method to filter out poor matches (outliers?). Lowe's ratio test is
            # a heuristic to select good matches between two sets of features. By asking
            # `knn_match` to find the 2 nearest neighbours, we can apply this test, which
            # compares the distance of the closest neighbour to the second closest
            # neighbour. The test asserts that if the ratio of the closest distance to
            # the second closest distance is below acertain threshold (e.g., 0.7), then
            # the match is considered good. The rational is that a good match is
            # significantly closer than the second best match. This helps to filter out
            # many false matches (FPs) where the difference between the best and second-
            # best is not significant.
            noisy_matches = self.matcher.knn_match(desc_query, desc_train, k=2)
            good_matches = self._perform_lowes_ratio_test(noisy_matches, lowe_ratio_test_threshold)

            # We require at least `min_match_count` good matches to be present to consider
            # this query image present in our train image.
            if len(good_matches) <= min_match_count:
                print(f"Skipping query image {img_path}. Not enough good matches found - " + \
                      f"{len(good_matches)}/{min_match_count}")
                continue

            try:
                print(f"Query image {img_path} yields enough good matches - " + \
                        f"{len(good_matches)}/{min_match_count}. Finding transform...")
                
                # Extract the locations of matched keypoints in both images.
                src_points = np.float32([kp_query[m.queryIdx].pt for m in good_matches]
                ).reshape(-1,1,2)
                dst_points = np.float32([kp_train[m.trainIdx].pt for m in good_matches]
                ).reshape(-1,1,2)

                # `findHomography` needs at least four correct points to find the
                # perspective transformation.
                M, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # M, mask = self.ransac.find_homography(src_points, dst_points)
                # matches_mask = None # FIXME

                # Define the corner points of the query image.
                h, w, _ = query_img.shape
                query_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                
                # Apply the homography matrix to transform the corner points of the
                # query image to the co-ordinate system of the train image.
                destination = cv.perspectiveTransform(query_corners, M)
                oriented_bounding_points = np.int32(destination)

                if draw:
                    polyline_train_img = cv.polylines(
                    train_img.copy(), [oriented_bounding_points], True, 255, 3, cv.LINE_AA)
                    
                    matches_img = cv.drawMatches(
                        query_img, kp_query, polyline_train_img, kp_train, good_matches, 
                        None, (0, 255, 0), None, matches_mask, 2)
                    
                    plt.imshow(matches_img)
                    plt.show()
            except:
                print(f"An exception was raised while handling query image {img_path}")
                continue

            axis_aligned_bbox = BoundingBox(oriented_bounding_points).align_with_axis()
            detection = Detection(img_path, axis_aligned_bbox)

            detections.append(detection)

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