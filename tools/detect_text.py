import time

import cv2
from loguru import logger

import configs.config as config
from models.craft_text_detector import Craft

if __name__ == "__main__":
    t0 = time.time()

    image = cv2.imread(config.PATH_IMAGE)

    craft = Craft(crop_type="box", cuda=False)

    crop_list, regions, image_detect = craft.detect_text(image)
    logger.info("Inference take {} s".format(time.time() - t0))
