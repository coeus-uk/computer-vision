import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from cv2.typing import MatLike
from typing import Iterator

MATCH_THRESHOLD = 0.8

def predict_bounding_boxes_cv2(img: MatLike, templates: list[np.ndarray], 
                            template_bounds: list[np.ndarray]) -> None:
    error = np.empty(len(templates))

    # Calculate bounding boxes of templates on image they are defined
    for (i, (template, bounds)) in enumerate(zip(templates, template_bounds)):
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # Calculate bounding box
        _, _, _, pred_top_left = cv2.minMaxLoc(result)
        height, width = template.shape
        pred_left, pred_top = pred_top_left
        pred_right = pred_left + width
        pred_bot = pred_top + height
        pred_bot_right = (pred_right, pred_bot)

        # Draw a rectangle around the matched region
        cv2.rectangle(img, pred_top_left, pred_bot_right, (0, 255, 0), 1)

        # Evaluate against annotation
        preds = np.array([pred_top, pred_bot, pred_left, pred_right])
        errors = np.abs(preds - bounds)
        error[i] = np.sum(errors)
        print(f"template {i + 1} | errors [top, bot, left, right] = {errors}")

def predict_bounding_boxes_pyramid_cv2(img_pyr: list[MatLike], 
                                       template_pyrs: list[list[MatLike]],
                                       bounds: list[np.ndarray],
                                       dst: MatLike) -> None:
    
    print(len(template_pyrs))
    print(len(bounds))
    # print(img_pyr[0].shape)
    # print(len(template_pyrs[0]))
    assert len(template_pyrs) == len(bounds)
    assert len(img_pyr) == len(template_pyrs[0])
    
    error = np.empty(len(template_pyrs))
    for (i, (template_pyr, t_bounds)) in enumerate(zip(template_pyrs, bounds)):
        for (img, template) in zip(img_pyr, template_pyr):
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            matches = np.where(result >= MATCH_THRESHOLD)
            if (matches[0].size > 0):
                # return matches
                _, _, _, pred_top_left = cv2.minMaxLoc(result)
                height, width = template.shape
                pred_left, pred_top = pred_top_left
                pred_right = pred_left + width
                pred_bot = pred_top + height
                pred_bot_right = (pred_right, pred_bot)

                # Draw a rectangle around the matched region
                cv2.rectangle(dst, pred_top_left, pred_bot_right, (0, 255, 0), 1)

                # Evaluate against annotation
                preds = np.array([pred_top, pred_bot, pred_left, pred_right])
                errors = np.abs(preds - t_bounds)
                error[i] = np.sum(errors)
                print(f"template {i + 1} | errors [top, bot, left, right] = {errors}")

                break


def laplacian_pyramid(img: MatLike, num_levels: int) -> list[MatLike]:
    """
    Builds an estimate for the laplacian pyramid of an image by using
    the differences of Gaussian.
    Modifies `img` inplace, a copy should be passed in if that is not desired.
    """
    # width, height = img.shape
    gauss_pyr = gaussian_pyrarmid(img, num_levels)
    laplace_pyr = []
    for i in range(num_levels - 1):
        a = gauss_pyr[i]
        b = cv2.pyrUp(gauss_pyr[i + 1])
        diff = cv2.subtract(a, b)
        laplace_pyr.append(diff)
    laplace_pyr.append(gauss_pyr[-1])

    return laplace_pyr

def gaussian_pyrarmid(img: MatLike, num_levels: int) -> list[MatLike]:
    """
    Build a gaussian pyramid for an image. 
    Modifies `img` inplace, a copy should be passed in if that is not desired.
    """
    pyramid = [img]
    for _ in range(num_levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)

    return pyramid

def images_with_annotations(dataset_folder: Path
                            ) -> Iterator[tuple[MatLike, pd.DataFrame]]:
    """
    Provides an iterator over the image dataset.
    It yields a tuple of the image along with the corresponding annotations.
    """
    annotations_folder = Path(dataset_folder, "annotations")
    image_paths = Path(dataset_folder, "images").iterdir()
    images_paths_sorted = sorted(image_paths, key=path_lexigraphical_order)

    for path in images_paths_sorted:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        csv_filename = Path(annotations_folder, path.stem).with_suffix(".csv")
        annotations = pd.read_csv(csv_filename)
        yield (img, annotations)

def path_lexigraphical_order(path: Path) -> tuple[int, str]:
    """
    Sorts paths first by length, then by alphabetical order.
    Returns a rich comparison object over Path objects. 
    Can be used as a sort delegate for inbuilt sort functions.
    """
    path_str = str(path)
    return (len(path_str), path_str)