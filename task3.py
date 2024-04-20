import logging
import cv2 as cv
import pandas as pd
import numpy as np

from enum import Enum
from pathlib import Path
from cv2 import DMatch, SIFT
from cv2.typing import MatLike
from dataclasses import dataclass
from typing_extensions import Self
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Tuple, List, Callable, Dict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]::[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def squared_error_loss(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Calculates the point-wise squared difference of two vectors.
    """
    return np.sum((truth - prediction) ** 2, axis=0)

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two vectors.
    """
    return np.linalg.norm(vec1 - vec2)


class Verbosity(Enum):

    HIGH = 2
    MEDIUM = 1
    LOW = 0


class ImageDataset:
    """
    Dataset of images with lazy loading.

    Iterates over images in a given directory, loading them one-by-one as needed.
    """
    def __init__(self, img_dir: Path, file_ext: str):
        self.img_dir = img_dir
        self.img_paths = list(img_dir.glob(f'*.{file_ext}'))
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

            return cv.imread(str(img_path), cv.IMREAD_COLOR), img_path
        else:
            raise StopIteration
        
    def print(self, i: int):
        assert 0 <= i < len(self.img_paths)
        img_path = self.img_paths[i]
        
        plt.imshow(cv.imread(str(img_path), cv.IMREAD_COLOR))
        plt.show()


class BruteForceMatcher:

    def __init__(self, distance_metric: Callable[[np.ndarray, np.ndarray], float]):
        self.distance = distance_metric

    def knn_match(self, src_vecs: np.ndarray, dst_vecs: np.ndarray, k: int) -> List[Tuple[DMatch, ...]]:
        matches = []

        # Ensure there are at least k destination vectors for each source vector.
        if len(dst_vecs) < k:
            return matches
        
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
    def fit(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, src: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass


class RANSAC(IModel):

    def __init__(
            self, 
            model_cls: IModel,
            model_hyperparams: dict,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
            min_datapoints: int,
            max_iterations: int = 14,
            threshold: float = 5.0,
            ):
        self.min_datapoints = min_datapoints
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.model_cls = model_cls
        self.loss = loss
        self.model_hyperparams = model_hyperparams
        self.best_model: IModel = None
        self.max_inliers = 0

    def fit(self, src: np.ndarray, dst: np.ndarray) -> Tuple[Self, None]:
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

        # TODO: Return a mask of the outliers
        return self, ()

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
        unnormalised = self.homography @ src
        normalised = unnormalised / unnormalised[2]

        return normalised
    
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

    top: int
    left: int
    bottom: int
    right: int
    
    @classmethod
    def from_series(cls: Self, series: pd.Series) -> Self:
        """
        Create an axis-aligned bounding box from a pandas series.
        """
        return cls(series.top, series.left, series.bottom, series.right)
    
    def compute_iou(self, other: Self) -> float:
        x_left = max(self.left, other.left)
        y_top = max(self.top, other.top)
        x_right = min(self.right, other.right)
        y_bottom = min(self.bottom, other.bottom)

        # Ensure there is an intersection, otherwise return nothing.
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        self_area = (self.right - self.left) * (self.bottom - self.top)
        other_area = (other.right - other.left) * (other.bottom - other.top)

        # Compute IoU
        return intersection_area / (self_area + other_area - intersection_area)


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

        axis_aligned_points = [min(x), min(y), max(x), max(y)]
        return AlignedBoundingBox(*axis_aligned_points)


@dataclass
class Detection:

    img_path: Path
    bounding_box: BoundingBox | AlignedBoundingBox

    @property
    def icon_name(self) -> str:
        return self.img_path.stem.split("-")[-1]


class ObjectDetector:
    """
    Object detector using SIFT keypoint localisation and descriptors.
    """
    def __init__(self, query_images: ImageDataset, sift_hyperparams: dict, ransac_hyperparams: dict, verbose: bool = True):
        self.query_images = query_images

        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        self.sift = SIFT.create(**sift_hyperparams)
        self.matcher = BruteForceMatcher(euclidean_distance)
        self.ransac = RANSAC(DirectLinearTransformer, {}, squared_error_loss, 4, **ransac_hyperparams)

    def detect(
            self, 
            train_img: MatLike, 
            lowe_ratio_test_threshold: float = 0.7, 
            min_match_count: int = 10,
            draw: bool = True
            ) -> Dict[str, Detection]:
        
        detections = {}
        kp_train, desc_train = self.sift.detectAndCompute(train_img, None)

        # If no keypoints or descriptors are detected, skip.
        if kp_train == () or desc_train is None:
            return detections

        for query_img, img_path in self.query_images:
            # Extract keypoints and generate descriptors using SIFT.
            kp_query, desc_query = self.sift.detectAndCompute(query_img, None)

            # If no keypoints or descriptors are detected, skip.
            if kp_query == () or desc_query is None:
                continue

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
                logger.info(f"Skipping query image {img_path}. Not enough good matches found - " + \
                      f"{len(good_matches)}/{min_match_count}")
                continue

            try:
                logger.info(f"Query image {img_path} yields enough good matches - " + \
                        f"{len(good_matches)}/{min_match_count}. Finding transform...")
                
                # Extract the locations of matched keypoints in both images and convert
                # them to homogeneous co-ordinates.
                src_points = np.float32([kp_query[m.queryIdx].pt + (1,) for m in good_matches]).T
                dst_points = np.float32([kp_train[m.trainIdx].pt + (1,) for m in good_matches]).T

                # Use RANSAC to reject outliers and estimate a homography for the
                # the remaining sets of inliers.
                self.ransac.reset()
                model, mask = self.ransac.fit(src_points, dst_points) 
                matches_mask = None # TODO: Write matches_mask

                # Define the corner points of the query image.
                h, w, _ = query_img.shape
                query_corners = np.float32([[0,0,1],[0,h-1,1],[w-1,h-1,1],[w-1,0,1]]).T
                
                # Apply the homography matrix to transform the corner points of the
                # query image to the co-ordinate system of the train image. Drop the
                # homogeneous row.
                destination = model.transform(query_corners)[:2]
                oriented_bounding_points = np.int32(destination).T.reshape(-1, 1, 2)

                if draw:
                    polyline_train_img = cv.polylines(
                    train_img.copy(), [oriented_bounding_points], True, 255, 3, cv.LINE_AA)
                    
                    matches_img = cv.drawMatches(
                        query_img, kp_query, polyline_train_img, kp_train, good_matches, 
                        None, (0, 255, 0), None, matches_mask, 2)
                    
                    plt.imshow(matches_img)
                    plt.show()
            except Exception as e:
                logger.warning(f"An exception was raised while handling query image {img_path}.\n{e}")
                continue

            axis_aligned_bbox = BoundingBox(oriented_bounding_points).align_with_axis()
            detection = Detection(img_path, axis_aligned_bbox)

            detections[detection.icon_name] = detection

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
            # first / second < lowe => first < lowe * second
            if first.distance < lowe_ratio_test_threshold * second.distance 
        ]
    

def evaluate_detections(detections: Dict[str, Detection], annotations: pd.DataFrame, iou_threshold: float = 0.5):
    tp, fp, fn = 0, 0, 0
    num_annotations = len(annotations)

    for _, gt in annotations.iterrows():
        if gt.classname not in detections:
            fn += 1
            continue

        detection = detections[gt.classname]

        gt_bbox = AlignedBoundingBox.from_series(gt)
        pred_bbox = detection.bounding_box

        iou = pred_bbox.compute_iou(gt_bbox)
        if iou > iou_threshold:
            tp += 1
        else:
            fp += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    fpr_denominator = (fp + num_annotations - tp)
    fpr = fp / fpr_denominator if fpr_denominator > 0 else 0
    accuracy = tp / num_annotations if num_annotations > 0 else 0

    return accuracy, tpr, fpr, fnr