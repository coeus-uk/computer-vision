import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from cv2.typing import MatLike
from typing import Iterator

def find_bounding_boxes_cv2(img: MatLike, templates: list[np.ndarray], 
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