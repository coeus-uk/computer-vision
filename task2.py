import pandas as pd
import cv2
import numpy as np
from scipy.signal import fftconvolve
from pathlib import Path
from copy import deepcopy
from cv2.typing import MatLike
from typing import Iterator

MATCH_THRESHOLD = 0.8

def predict_all_templates(templates: list[tuple[str, list[MatLike]]],
                          img: MatLike, 
                          annotations: pd.DataFrame,
                          dst: MatLike | None = None,
                          num_pyramid_levels: int = 3) -> None:
    """
    Predict bounding boxes for a number of templates in a given image.
    Calculate the error if a match is found.
    """
    img_pyramid = laplacian_pyramid(deepcopy(img), num_pyramid_levels)
    bounding_boxes = { row.classname: np.array([row.left, row.right, row.top, row.bottom])
                      for row in annotations.itertuples() }
    
    for (classname, template_pyr) in templates:
        pred_bounds = predict_bounding_box(img_pyramid, template_pyr, dst)

        if pred_bounds is None and classname not in bounding_boxes: 
            print(f"{classname}\t| PASS | Correctly didn't match template")
            continue

        if pred_bounds is None and classname in bounding_boxes:
            print(f"{classname}\t| FAIL | Didn't match template when it existed")
            continue

        annotated_bounds = bounding_boxes[classname]
        if annotated_bounds is None:
            print(f"{classname}\t| FAIL | Predicted template is not in the image. "
                  + f"Predicted bounds: {pred_bounds}")
            continue
        
        errors = np.abs(pred_bounds - annotated_bounds)
        print(f"{classname}\t| PASS | errors [top, bot, left, right] ="
                + f" {errors} | total error = {np.sum(errors)}")

def predict_bounding_box(img_pyr: list[MatLike], 
                         template_pyr: list[MatLike],
                         dst: MatLike | None) -> np.ndarray | None:
    """
    Predict the bounding box of a template in an image.

    Return None when there is no match, otherwise return an array 
    of positions making up the bounding box [top, bot, left, right]
    """

    assert len(img_pyr) == len(template_pyr)
    
    for (img, template) in zip(img_pyr, template_pyr):
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) # <-- this is probably not allowed
        matches = np.where(result >= MATCH_THRESHOLD)
        if (matches[0].size > 0):
            _, _, _, pred_top_left = cv2.minMaxLoc(result)
            height, width = template.shape
            pred_left, pred_top = pred_top_left
            pred_right = pred_left + width
            pred_bot = pred_top + height
            preds = np.array([pred_top, pred_bot, pred_left, pred_right])

def correlate_ssd(img: np.ndarray, template: np.ndarray
                 ) -> np.ndarray:
    """
    Calculate sum square difference along a sliding window 
    between an image and kernel.
    """
    # ssd = (a1 - a2)^2 + (b1 - b2)^2 + ...
    # Recall that (a1 - a2)^2 = a1^2 + a2^2 - 2(a1)(a2)
    # So, ssd = a1^2 + a2^2 - 2(a1)(a2) + b1^2 + b2^2 - 2(b1)(b2) + ...
    #         = (a1^2 + b1^2 + ...) + (a2^2 + b2^2 + ...) - 2 * ((a1)(a2) + (b1)(b2) + ...)
    # (a1^2 + b1^2 + ...) is a sliding sum over the square of the image
    #  which is equivalent to a convolution of the image using a kernel of 1s
    # (a2^2 + b2^2 + ...) is the sum of the square of template
    # ((a1)(a2) + (b1)(b2) + ...) is the cross correlation of the template over the image
    #  which is equivalent to the convolution of an inverted template over the image

    # Thus ssd can be calculated as below.
    sliding_sum_of_squares = fftconvolve(np.square(img), np.ones(template.shape), mode='valid')
    cross_correlation = fftconvolve(img, template[::-1, ::-1], mode='valid')
    sum_of_template_squares = np.sum(np.square(template))
    ssd = sliding_sum_of_squares  + sum_of_template_squares - 2 * cross_correlation

    return ssd

def correlate_ssd_normed(img: np.ndarray, template: np.ndarray
                        ) -> np.ndarray:
    """
    Calculate sum square difference along a sliding window 
    between an image and kernel.
    """
    norm_image = (img - np.mean(img)) / np.std(img)
    norm_template = (template - np.mean(template)) / np.std(template)
    return correlate_ssd(norm_image, norm_template)

def laplacian_pyramid(img: MatLike, num_levels: int) -> list[MatLike]:
    """
    Builds an estimate for the laplacian pyramid of an image by using
    the differences of Gaussian.
    Modifies `img` inplace, a copy should be passed in if that is not desired.
    """
    gauss_pyr = gaussian_pyrarmid(img, num_levels)
    neighbouring_layers = [(gauss_pyr[i], cv2.pyrUp(gauss_pyr[i + 1])) 
                           for i in range(num_levels - 1)]
    diff_of_gauss = [cv2.subtract(a, b) for (a, b) in neighbouring_layers]
    diff_of_gauss.append(gauss_pyr[-1])
    return diff_of_gauss

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