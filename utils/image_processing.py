import math

import numpy as np
import torch
from PIL import Image


def resize(w, h, expected_height, img_min_width, img_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, img_min_width)
    new_w = min(new_w, img_max_width)

    return new_w, expected_height


def ocr_input_processing(imgs, img_height, img_min_width, img_max_width):
    ##########################################################
    # This function can process multiple image to form a batch
    # -input: an pillow image
    #         or list of pillow image with same size(WxH)
    # -output: torch tensor (len(imgs)xCxW'xH')
    ##########################################################
    processed_img = []
    if not isinstance(imgs, list):
        imgs = [imgs]
    for img in imgs:
        w, h = img.size
        new_w, img_height = resize(w, h, img_height, img_min_width, img_max_width)
        img = img.resize((new_w, img_height), Image.ANTIALIAS)
        img = np.asarray(img).transpose(2, 0, 1)
        img = img / 255
        img = img[np.newaxis, ...]
        processed_img.append(img)

    final_inputs = torch.FloatTensor(np.concatenate(processed_img))
    return final_inputs
