import pandas as pd
import cv2
import numpy as np
from scipy.signal import fftconvolve
from pathlib import Path
from cv2.typing import MatLike
from typing import Iterator

TEMPLATE_PYRAMID_SCALES = [5/8, 4/8, 3/8, 2/8, 1/8]

CCOEFF_NORMED_THRESHOLD = 0.8

THRESHOLD_SCALE = 0.8

# 70000,
# 2000,
# 2000,
# 500,
# 100]
THRESHOLDS = [
    10000000,
    5000000,
    1500000,
    165000,
    100000]
# THRESHOLDS = [
#     5.583934789155553e+18,
#     4.315835850200041e+17,
#     3.199918135465214e+16,
#     2178193442070275.0,
#     138047930383868.83
# ]
# THRESHOLDS = [
#     0.95,
#     0.93,
#     0.9,
#     0.85,
#     0.8]

def predict_all_templates(img_pyr: list[MatLike], 
                        templates: list[tuple[str, list[MatLike]]]
                        ) -> tuple[list[str], list[float], list[np.ndarray]]:
    """
    Predict bounding boxes for a number of templates in a given image.
    Calculate the error if a match is found.
    """
    
    classnames, scores, pred_boxes = [], [], []
    for classname, template_pyr in templates:
        matches = predict_bounding_box(img_pyr, template_pyr, classname)
        first_match = next(matches, None)
        if (first_match is None): continue
        score, pred_box = first_match
        classnames.append(classname)
        scores.append(score)
        pred_boxes.append(pred_box)
    
    return classnames, scores, pred_boxes
       

def predict_bounding_box(img_pyr: list[MatLike], template_pyr: list[MatLike], 
                         classname: str, thresholds: list[int] = THRESHOLDS) -> Iterator[tuple[float, np.ndarray]]:
    """
    Use multi resolution template matching to match a given collection of 
    tempaltes in an image. 
    Returns the matches predicted bounding box as well as its score.
    """
    for t_level, template_layer in enumerate(template_pyr):
        template_h, template_w = template_layer.shape
        for i_level, img_layer in enumerate(img_pyr):
            img_h, img_w = img_layer.shape

            # Skip levels where the template is larger than the image
            if template_h > img_h or template_w > img_w: continue 
        
            # Perform template matching
            result = cv2.matchTemplate(img_layer, template_layer,
                                        cv2.TM_CCOEFF_NORMED) # <- this isn't allowed
            # result = cv2.matchTemplate(img_layer, template_layer, cv2.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            score = max_val
            pred_top_left = max_loc
            if score > CCOEFF_NORMED_THRESHOLD:#thresholds[t_level]:
                img_sf = 2 ** i_level
                left, top = pred_top_left
                left, top = (int(left * img_sf), int(top * img_sf))
                right = left + template_w
                bot = top + template_h
                pred_box = np.array([top, bot, left, right])
                # print(f"Match found for {classname} at level: {i_level} with score: {score} > 0.8, preds = {pred_box}")
                yield (score, pred_box)

def non_max_suppression(preds: list[np.ndarray], scores: list[float],
                        classnames: list[str]) -> None:
    """
    Applies non-maximal suppression to a list of bounding boxes.
    Detects overlapping bounding boxes and discards the one with a lower score.
    """
    assert len(preds) == len(scores)
    
    # Tolerance for very slight overlaps
    iou_threshold = 0.005 

    boxes_to_remove = set()
    for j, box in enumerate(preds):
        for i, other_box in enumerate(preds):
            if i == j: continue
            iou = calculate_iou(box, other_box)
            if (iou > iou_threshold and scores[i] < scores[j]):
                boxes_to_remove.add(i)

    for i in sorted(boxes_to_remove, reverse=True):
        preds.pop(i)
        scores.pop(i)
        classnames.pop(i)

def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate the intersection over area for two given bounding boxes
    """

    top_a, bot_a, left_a, right_a = box_a
    top_b, bot_b, left_b, right_b = box_b

    intersection_top = max(top_a, top_b)
    intersection_left = max(left_a, left_b)
    intersection_bot = min(bot_a, bot_b)
    intersection_right = min(right_a, right_b)

    intersection_width = intersection_right - intersection_left
    intersection_height = intersection_bot - intersection_top
    intersection_area = max(intersection_width, 0) * max(intersection_height, 0)

    total_area_a = (right_a - left_a) * (bot_a - top_a)
    total_area_b = (right_b - left_b) * (bot_b - top_b)

    union_area = total_area_a + total_area_b - intersection_area
    return intersection_area / union_area

def annotate_predictions(img: MatLike, classnames: list[str],
                         scores: list[int], pred_boxes: np.ndarray) -> None:
    """
    Draw a given collection of bounding boxes onto a destination image. 
    """
    black: cv2.Scalar = (0, 255, 0)
    for classname, score, pred_box in zip(classnames, scores, pred_boxes):
        top_left = (pred_box[2], pred_box[0])
        bot_right = (pred_box[3], pred_box[1])
        cv2.rectangle(img, top_left, bot_right, color=black, thickness=1)
        cv2.putText(img, f"{classname}({score:.2f})", top_left, 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black)

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

def correlate_ssd_normed(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Calculate sum square difference along a sliding window 
    between an image and kernel.
    """
    norm_image = (img - np.mean(img)) / np.std(img)
    norm_template = (template - np.mean(template)) / np.std(template)
    return correlate_ssd(norm_image, norm_template)

import numpy as np
import scipy.fftpack as fp

def phase_correlation(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    norm_img = (img - np.mean(img)) / np.std(img)
    norm_template = (template - np.mean(template)) / np.std(template)

    img_ft = np.fft.rfft2(norm_img)                   
    template_ft = np.fft.rfft2(norm_template, s=img.shape)

    cross_correlation_ft = img_ft * np.conj(template_ft)
    norm_cc_ft = cross_correlation_ft / np.abs(cross_correlation_ft)
    
    norm_cc = np.fft.irfft2(norm_cc_ft)
    return norm_cc


def template_pyramid_by_classname(dataset_folder: Path,
                                  file_extension: str = ".png"
                                  ) -> Iterator[tuple[str, list[MatLike]]]:
    """
    Load templates found in a given directory and return an iteratior over 
    grouped classnames and template gaussian pyramids
    """
    image_paths = dataset_folder.glob(f"*{file_extension}")
    images_paths_sorted = sorted(image_paths)
    for path in images_paths_sorted:
        template = cv2.imread(str(path))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Gaussian pyramid with custom scaling
        width, height = template.shape
        template_pyr = []
        for scale in TEMPLATE_PYRAMID_SCALES:
            new_width = int(width * scale)
            new_height = int(height * scale)
            template = cv2.resize(template,
                                          (new_width, new_height),
                                          interpolation=cv2.INTER_AREA)
            template_pyr.append(template)
        
        # Annotations are prefixed by 2 numbers, the given dataset has 3.
        classname = path.stem[1:] 

        yield (classname, template_pyr)

def laplacian_pyramid(img: MatLike, num_levels: int) -> list[MatLike]:
    """
    Builds an estimate for the laplacian pyramid of an image by using
    the difference of Gaussian method.
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
                            ) -> Iterator[tuple[MatLike, list[tuple[str, np.ndarray]]]]:
    """
    Provides an iterator over the image dataset.
    It yields a tuple of the image along with the corresponding
    annotated bounding boxes.
    """
    annotations_folder = Path(dataset_folder, "annotations")
    image_paths = Path(dataset_folder, "images").iterdir()
    images_paths_sorted = sorted(image_paths, key=path_lexigraphical_order)

    for path in images_paths_sorted:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        csv_filename = Path(annotations_folder, path.stem).with_suffix(".csv")
        annotations = pd.read_csv(csv_filename)
        template_bounds = [(row.classname, np.array([row.left, row.right, row.top, row.bottom])) 
                           for row in annotations.itertuples()]
        yield (img, template_bounds)

def path_lexigraphical_order(path: Path) -> tuple[int, str]:
    """
    Sorts paths first by length, then by alphabetical order.
    Returns a rich comparison object over Path objects. 
    Can be used as a sort delegate for inbuilt sort functions.
    """
    path_str = str(path)
    return (len(path_str), path_str)

def evaluation_metrics(predicted_bounds: dict[str, np.ndarray], 
                       annotated_bounds: list[tuple[str, np.ndarray]],
                       pred_threshold: float = 0.95
                       ) -> tuple[float, float, float, float]:
    """
    Calculates the accuracy, true positive rate, false positive rate,
    and false negative rate given a collection of predicted bounding boxes 
    and a colleciton of expected results.
    """
    num_tp, num_fp, num_fn = 0, 0, 0
    num_boxes = len(annotated_bounds)

    for classname, bound in annotated_bounds:
        if classname not in predicted_bounds:
            num_fn += 1
            continue

        pred_box = predicted_bounds[classname]
        iou = calculate_iou(pred_box, bound)
        if iou >= pred_threshold:
            num_tp += 1
        else:
            num_fp += 1

    accuracy = num_tp / num_boxes if num_boxes != 0 else 0
    true_pos_rate = num_tp / (num_tp + num_fn) if (num_tp + num_fn) != 0 else 0
    fpr_denominator = (num_fp + num_boxes - num_tp)
    false_pos_rate = num_fp / fpr_denominator if fpr_denominator != 0 else 0
    false_neg_rate = num_fn / (num_fn + num_tp) if (num_fn + num_tp) != 0 else 0
    
    return accuracy, true_pos_rate, false_pos_rate, false_neg_rate