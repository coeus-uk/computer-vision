import pandas as pd
import cv2
import numpy as np
from scipy.signal import fftconvolve, convolve, correlate
from pathlib import Path
from cv2.typing import MatLike
from typing import Iterator

TEMPLATE_PYRAMID_SCALES = [5/8, 4/8, 3/8, 2/8, 1/8]

def predict_all_templates(img_pyr: list[MatLike],
                          templates: list[tuple[str, list[MatLike]]]
                          ) -> tuple[list[str], list[float], list[np.ndarray]]:
    """
    Predict bounding boxes for a number of templates in a given image.
    Calculate the error if a match is found.
    """
    
    classnames, scores, pred_boxes, results = [], [], [], []
    for classname, template_pyr in templates:
        matches = predict_bounding_box(img_pyr, template_pyr)
        all_matches = [match for match in matches]
        if (len(all_matches) == 0): continue

        all_matches.sort(key=lambda m: m[0], reverse=True)
        score, pred_box = all_matches[0]
        classnames.append(classname)
        scores.append(score)
        pred_boxes.append(pred_box)
    
    return classnames, scores, pred_boxes
       

def predict_bounding_box(img_pyr: list[MatLike], template_pyr: list[MatLike]
                         ) -> Iterator[tuple[float, np.ndarray]]:
    """
    Use multi resolution template matching to match a given collection of 
    tempaltes in an image. 
    Returns the matches predicted bounding box as well as its score.
    """
    for template_layer in template_pyr:
        template_h, template_w = template_layer.shape
        for i_level, img_layer in enumerate(img_pyr):
            img_h, img_w = img_layer.shape

            # Skip levels where the template is larger than the image
            if template_h > img_h or template_w > img_w: continue 
        
            # Perform template matching
            # result = cv2.matchTemplate(norm_image, norm_template,
            #                             cv2.TM_SQDIFF_NORMED) # <- this isn't allowed
            # set background to black
            # img_layer[img_layer > 250] = 0s
            result = phase_correlation(img_layer, template_layer)
            # result = correlate_ssd_normed2(img_layer, template_layer)
            # result = cv2.matchTemplate(img_layer, template_layer, cv2.TM_CCOEFF_NORMED)
            # result = cv2.matchTemplate(img_layer, template_layer, cv2.TM_SQDIFF_NORMED)
            # result = correlate_naive(img_layer, template_layer)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            score = max_val
            pred_top_left = max_loc
            if score > 0.0:
                img_sf = 2 ** i_level
                left, top = pred_top_left
                left, top = (int(left * img_sf), int(top * img_sf))
                right = left + (template_w * img_sf)
                bot = top + (template_h * img_sf)
                pred_box = np.array([top, bot, left, right])
                # print(f"Match found for {classname} at level: {i_level} with score: {score} > 0.8, preds = {pred_box}")
                yield (score, pred_box)

def non_max_suppression(preds: list[np.ndarray], scores: list[float],
                        classnames: list[str], iou_threshold: float = 0.005
                        ) -> None:
    """
    Applies non-maximal suppression to a list of bounding boxes.
    Detects overlapping bounding boxes and discards the one with a lower score.
    """
    assert len(preds) == len(scores)
    
    eval_order = np.argsort(np.array(scores))[::-1]
    boxes_to_remove = set()

    for j in eval_order:
        box = preds[j]
        for i in eval_order[::-1]:
            other_box = preds[i]
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
                         scores: list[int], pred_boxes: list[np.ndarray]
                        ) -> None:
    """
    Draw a given collection of bounding boxes onto a destination image. 
    """
    black: cv2.Scalar = (0, 0, 0)
    for classname, score, pred_box in zip(classnames, scores, pred_boxes):
        top_left = (pred_box[2], pred_box[0])
        bot_right = (pred_box[3], pred_box[1])
        cv2.rectangle(img, top_left, bot_right, color=black, thickness=1)
        cv2.putText(img, f"{classname}({score:.2f})", top_left, 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black)

def correlate_naive(img: np.ndarray, template: np.ndarray
                    ) -> np.ndarray: 
    """
    2D discrete correlation is simply 2D discrete convolution with a flipped template.
    """
    result = convolve(img, template[::-1, ::-1], mode='valid')
    # result = cv2.normalize(result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return result

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

    return ssd, sliding_sum_of_squares

def correlate_ssd_normed(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Calculate sum square difference along a sliding window 
    between an image and kernel.
    """
    norm_image = (img - np.mean(img)) / np.std(img)
    norm_template = (template - np.mean(template)) / np.std(template)
    ssd = correlate_ssd(norm_image, norm_template)

    return ssd

def correlate_ssd_normed2(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    # norm_image = img - np.mean(img)) / np.std(img)
    # norm_template = (template - np.mean(template)) / np.std(template)

    # ssd, sliding_sum_of_squares = correlate_ssd_normed(img, template)

    # norm_factor = np.sqrt(sliding_sum_of_squares * np.sum(np.square(template)))

    # return ssd / norm_factor
    img = img + 1
    template = template + 1

    image_height, image_width = img.shape
    template_height, template_width = template.shape
    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1
    normed_ssd_result = np.zeros((result_height, result_width), dtype=np.float32)
    template_sq_sum = np.sum(np.square(template))
    
    for y in range(result_height):
        for x in range(result_width):
            window = img[y:y+template_height, x:x+template_width]
            window_sq_sum = np.sum(np.square(window))
            diff = np.subtract(window, template)

            denom = np.sqrt(window_sq_sum * template_sq_sum)

            numerator = np.sum(np.square(diff))
            normed_ssd_result[y, x] = numerator / denom    
    return normed_ssd_result

def phase_correlation(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    norm_img = (img - np.mean(img)) / np.std(img)
    norm_template = (template - np.mean(template)) / np.std(template)

    img_ft = np.fft.rfft2(norm_img)                   
    template_ft = np.fft.rfft2(norm_template, s=img.shape)

    cross_correlation_ft = img_ft * np.conj(template_ft)
    norm_cc_ft = cross_correlation_ft / np.abs(cross_correlation_ft)
    
    norm_cc = np.fft.irfft2(norm_cc_ft)
    return norm_cc

def normed_cross_correlation(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    """

    mean_filter = np.ones_like(template) / (template.size)
    img_means = convolve(img, mean_filter, mode='same')
    img_0mean = img - img_means
    template_0mean = template - np.mean(template)

    # img_std  = np.std (img)
    img_std_dvns = convolve(np.square(img_0mean), mean_filter, mode='valid')
    img_std_dvns = np.sqrt(np.abs(img_std_dvns))
    template_std_dvns = np.std(template)

    numerator = correlate(img_0mean, template_0mean, mode='valid')
    denominator = img_std_dvns * template_std_dvns

    return numerator / denominator
    
    # sliding_sum_of_squares = fftconvolve(np.square(img), np.ones(template.shape), mode='valid')
    # cross_correlation = fftconvolve(img, template[::-1, ::-1], mode='valid')
    # sum_of_template_squares = np.sum(np.square(template))
    # ssd = sliding_sum_of_squares  + sum_of_template_squares - 2 * cross_correlation

    # templace_std  = np.std (template)
    
    # corr = correlate(img_0mean, template_0mean, mode = 'valid' )
    # corr = corr / (img.size * img_std * templace_std)
    # return corr



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
        # template[template > 250] = 0
        
        # icon_bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # alpha = icon_bgra[:, :, 3]
        # alpha_mask = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        # icon_bgr = cv2.cvtColor(icon_bgra, cv2.COLOR_BGRA2BGR)
        # icon_masked: np.ndarray = cv2.bitwise_and(icon_bgr, alpha_mask)
        # # icon_masked = cv2.copyMakeBorder(icon_masked, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        # template = cv2.cvtColor(icon_masked, cv2.COLOR_BGR2GRAY)

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
            template_pyr.append(cv2.rotate(template.copy(),
                                           cv2.ROTATE_90_CLOCKWISE))
            template_pyr.append(cv2.rotate(template.copy(),
                                           cv2.ROTATE_180))
            template_pyr.append(cv2.rotate(template.copy(),
                                           cv2.ROTATE_90_COUNTERCLOCKWISE))
        
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
                            ) -> Iterator[tuple[MatLike, dict[str, np.ndarray]]]:
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
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        csv_filename = Path(annotations_folder, path.stem).with_suffix(".csv")
        annotations = pd.read_csv(csv_filename)
        template_bounds = {row.classname:np.array([row.left, row.right, row.top, row.bottom]) 
                           for row in annotations.itertuples()}
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
                       annotated_bounds: dict[str, np.ndarray],
                       template_metrics: dict[str, np.ndarray],
                       pred_threshold: float = 0.5
                       ) -> tuple[float, float, float, float]:
    """
    Calculates the accuracy, true positive rate, false positive rate,
    and false negative rate given a collection of predicted bounding boxes 
    and a colleciton of expected results.
    """
    num_tp, num_fp, num_fn = 0, 0, 0
    num_boxes = len(annotated_bounds)

    for classname, bound in annotated_bounds.items():
        if classname not in predicted_bounds:
            num_fn += 1
            template_metrics[classname][2] += 1 
            continue

        pred_box = predicted_bounds[classname]
        iou = calculate_iou(pred_box, bound)
        if iou >= pred_threshold:
            template_metrics[classname][0] += 1 
            num_tp += 1
        else:
            template_metrics[classname][1] += 1 
            num_fp += 1

    for classname, bound in predicted_bounds.items():
        if classname not in annotated_bounds:
            num_fp += 1

    accuracy = num_tp / num_boxes if num_boxes != 0 else 0
    true_pos_rate = num_tp / (num_tp + num_fn) if (num_tp + num_fn) != 0 else 0
    fpr_denominator = (num_fp + num_tp)
    false_pos_rate = num_fp / fpr_denominator if fpr_denominator != 0 else 0
    false_neg_rate = num_fn / (num_fn + num_tp) if (num_fn + num_tp) != 0 else 0
    
    return accuracy, true_pos_rate, false_pos_rate, false_neg_rate

def rotate_bounding_box(dataset_folder: Path, image_size: tuple, angle: float
                        ) -> None:
    annotations_folder = Path(dataset_folder, "annotations")
    csv_filename = Path(annotations_folder, "test_image_1").with_suffix(".csv")
    annotations = pd.read_csv(csv_filename)
    modified_rows = []
    for row in annotations.itertuples():
        left = row.top
        top = row.left
        right = row.bottom
        bottom = row.right

        width, height = image_size
        centre = (width // 2, height // 2)
        angle_rad = np.deg2rad(angle)

        right, bottom = rotate_point((right, bottom), centre, angle_rad)
        left, top = rotate_point((left, top), centre, angle_rad)

        # row.left, row.right, row.top, row.bottom = top, bottom, left, right
        modified_rows.append((row.classname, left, top, right, bottom))

    updated_csv_path = Path(annotations_folder, f"test_image_1_{angle}").with_suffix(".csv")
    pd.DataFrame(data=modified_rows, columns=annotations.columns).to_csv(updated_csv_path)

        
def rotate_point(point: tuple, centre: tuple, angle_rad: float
                 ) -> tuple[float, float]:
    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)
    translated_point = (point[0] - centre[0], point[1] - centre[1])
    x_new = translated_point[0] * cos - translated_point[1] * sin
    y_new = translated_point[0] * sin + translated_point[1] * cos
    rotated_point = (x_new + centre[0], y_new + centre[1])
    return rotated_point

def histogram_normalise(image: MatLike) -> MatLike:
    # images = task2.images_with_annotations(Path("Task2Dataset"))
    # for i, (img, _) in enumerate(images):
    #     noise_image = histogram_normalise(img)
    #     cv2.imwrite(f"./Task2Dataset/coloured-images/test_image_{i+1}.png", noise_image)

    """Apply histogram normalisation to image."""
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return equalized_image

def add_gaussian_noise(image: MatLike, mean: float, sigma: float) -> MatLike:
    # images = task2.images_with_annotations(Path("Task2Dataset"))
    # for i, (img, _) in enumerate(images):
    #     noise_image = add_gaussian_noise(img, mean=0, sigma=10)
    #     cv2.imwrite(f"./Task2Dataset/coloured-images/test_image_{i+1}.png", noise_image)

    """Add Gaussian noise to the image."""
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def debug_template_metrics(template_metrics: dict[str, np.ndarray]) -> None:
    images = images_with_annotations(Path("Task2Dataset"))
    all_annotaitons = [annotations for (_, annotations) in images]
    occurences = {}
    total_number_of_templates = 0 
    for image_annotations in all_annotaitons:
        for classname, _ in image_annotations:
            if classname not in occurences:
                occurences[classname] = 1
            else:
                occurences[classname] += 1
            total_number_of_templates += 1

    classnames = np.array(list(template_metrics.keys()))
    true_positives = np.zeros(classnames.shape)
    false_positives = np.zeros(classnames.shape)
    false_negatives = np.zeros(classnames.shape)

    template_accuracies = np.zeros(classnames.shape)
    true_positive_rate = np.zeros(classnames.shape)
    false_positive_rate = np.zeros(classnames.shape)
    false_negative_rate = np.zeros(classnames.shape)

    for i, (classname, t_m) in enumerate(template_metrics.items()):
        if classname in occurences:
            tps, fps, fns = t_m[0], t_m[1], t_m[2]
            true_positives[i] = tps
            false_positives[i] = fps
            false_negatives[i] = fns

            print(f"{classname} | t_positives: {tps}; f_positives: {fps}, f_negatives: {fns}")
            accuracy = tps / occurences[classname]
            tpr = tps / (tps + fns)
            fpr = fps / (fps + total_number_of_templates - tps)
            fnr = fns / (fns + tps)

            template_accuracies[i] = accuracy * 100
            true_positive_rate[i] = tpr * 100
            false_positive_rate[i] = fpr * 100
            false_negative_rate[i] = fnr * 100

            print(f"{classname} | acc: {accuracy}; true pos rate: {tpr}; false pos rate {fpr}; false neg rate {fnr}")
        
    print(classnames)

    # write to csv
    columns = ["classname", "# true positives", "# false positives", "# false negatives"]
    data = [[classname, tp, fp, fn]
            for (classname, tp, fp, fn) in zip(classnames, true_positives, false_positives, false_negatives) if classname in occurences]
    template_classifications = pd.DataFrame(data=data, columns=columns)
    template_classifications.to_csv("task2-evaluation/classifications-per-template.csv")

    columns = ["classname", "accuracy", "true positive rate", "false positive rate", "false negative rate"]
    data = [[classname, accuracy, tpr, fpr, fnr] 
            for (classname, accuracy, tpr, fpr, fnr) in 
            zip(classnames, template_accuracies, true_positive_rate, false_positive_rate, false_negative_rate)
            if classname in occurences]
    template_metrics_df = pd.DataFrame(data=data, columns=columns)
    template_metrics_df.to_csv("task2-evaluation/metrics-per-template.csv")
    # template_metrics_df.plot(x="classname", y=["accuracy", "true positive rate", "false positive rate", "false negative rate"])
    # template_metrics_df.plot(x="classname", y="accuracy")
    # pd.DataFrame(dataframe, columns=columns).to_csv("metrics-per-template.csv")