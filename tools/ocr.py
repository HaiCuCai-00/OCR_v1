import os
import sys
import time

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from modules.ocr.text_classifier import TextClassifier
from utils.image_processing import ocr_input_processing

sys.path.append("../")


class OCR:
    def __init__(self, config_path="../configs/ocr/transformer_ocr.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self.text_classifier = TextClassifier(self.cfg)

    def get_result(self, imgs):
        ###############################################################
        # Func ocr_input_processing can process multiple image as batch.
        # We can do that by pass list of PIL image in the first argument.
        # However all the image in the list must have same size
        ################################################################
        final_res = []
        for img in imgs:
            final_inputs = ocr_input_processing(
                img,
                self.cfg.model.input.image_height,
                self.cfg.model.input.image_min_width,
                self.cfg.model.input.image_max_width,
            )
            t0 = time.time()
            res = self.text_classifier.predict(final_inputs)
            final_res.append([res, time.time() - t0])
        return final_res


if __name__ == "__main__":
    demo_image_dir = "../data/demo_data/"
    imgs = []
    for img_name in os.listdir(demo_image_dir):
        img = Image.open(os.path.join(demo_image_dir, img_name))
        imgs.append(img)
    ocr_module = OCR()
    final_results = ocr_module.get_result(imgs)
    for result in final_results:
        logger.info("RESULT: {} (INFERENCE TAKE {} s)".format(result[0], result[1]))
