import copy
import os

import cv2
import numpy as np

import configs.config as config
from models.craft_text_detector.image_utils import read_image


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
            ):
                img_files.append(os.path.join(dirpath, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dirpath, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dirpath, file))
            elif ext == ".zip":
                continue

    return img_files, mask_files, gt_files


def rectify_poly(img, poly):
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return output_img


def crop_poly(image, poly):
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    # crop around poly
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(poly)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    return cropped


def export_detected_region(image, poly, rectify=True):
    """
    Arguments:
        image: full image
        points: bbox or poly points
        file_path: path to be exported
        rectify: rectify detected polygon by affine transform
    """
    if rectify:
        # rectify poly region
        result = rectify_poly(image, poly)
    else:
        result = crop_poly(image, poly)

    # export corpped region
    return result


def scale_bounding_box(region):
    region = np.array(region).astype(np.int32).reshape((-1))

    region = region.reshape(-1, 2)

    # top left point
    region[0][0] = region[0][0] - config.CON_PADDING
    region[0][1] = region[0][1] - config.CON_PADDING

    # top right point
    region[1][0] = region[1][0] + config.CON_PADDING
    region[1][1] = region[1][1] - config.CON_PADDING

    # bottom right point
    region[2][0] = region[2][0] + config.CON_PADDING
    region[2][1] = region[2][1] + config.CON_PADDING

    # bottom left point
    region[3][0] = region[3][0] - config.CON_PADDING
    region[3][1] = region[3][1] + config.CON_PADDING

    return region


def export_detected_regions(
    image,
    regions,
    rectify: bool = False,
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        regions: list of bboxes or polys
        file_name (str): export image file name
        output_dir: folder to be exported
        rectify: rectify detected polygon by affine transform
    """

    # read/convert image
    image = read_image(image)

    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # init crop_list
    crop_list = []

    # export regions
    for ind, region in enumerate(regions):
        # get export path
        region = scale_bounding_box(region)
        image_crop = export_detected_region(image, poly=region, rectify=rectify)
        # note exported file path
        crop_list.append(image_crop)

    return crop_list


def export_extra_results(image, regions):
    for i, region in enumerate(regions):
        region = scale_bounding_box(region)
        detect = cv2.polylines(
            image,
            [region.reshape((-1, 1, 2))],
            True,
            color=(0, 0, 255),
            thickness=1,
        )

    return detect
