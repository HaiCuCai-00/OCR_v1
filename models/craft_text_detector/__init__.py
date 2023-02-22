from __future__ import absolute_import

import models.craft_text_detector.craft_utils as craft_utils
import models.craft_text_detector.file_utils as file_utils
import models.craft_text_detector.image_utils as image_utils
import models.craft_text_detector.predict as predict
import models.craft_text_detector.torch_utils as torch_utils

__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
]

read_image = image_utils.read_image
load_craftnet_model = craft_utils.load_craftnet_model
load_refinenet_model = craft_utils.load_refinenet_model
get_prediction = predict.get_prediction
export_detected_regions = file_utils.export_detected_regions
export_extra_results = file_utils.export_extra_results
empty_cuda_cache = torch_utils.empty_cuda_cache


class Craft:
    def __init__(
        self,
        output_dir=None,
        rectify=True,
        export_extra=True,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280,
        refiner=True,
        crop_type="box",
    ):

        self.craft_net = None
        self.refine_net = None
        self.output_dir = output_dir
        self.rectify = rectify
        self.export_extra = export_extra
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.refiner = refiner
        self.crop_type = crop_type

        # load craftnet
        self.load_craftnet_model()
        # load refinernet if required
        if refiner:
            self.load_refinenet_model()

    def load_craftnet_model(self):
        """
        Loads craftnet model
        """
        self.craft_net = load_craftnet_model()

    def load_refinenet_model(self):
        """
        Loads refinenet model
        """
        self.refine_net = load_refinenet_model()

    def unload_craftnet_model(self):
        """
        Unloads craftnet model
        """
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self):
        """
        Unloads refinenet model
        """
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(self, image, image_path=None):

        if image_path is not None:
            print("Argument 'image_path' is deprecated, use 'image' instead.")
            image = image_path

        if self.cuda:
            self.refine_net.cuda()
            self.craft_net.cuda()

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.cuda,
            long_size=self.long_size,
        )

        # arange regions
        if self.crop_type == "box":
            regions = prediction_result["boxes"]
        elif self.crop_type == "poly":
            regions = prediction_result["polys"]
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")

        crop_list = export_detected_regions(
            image=image,
            regions=regions,
            rectify=self.rectify,
        )

        result = export_extra_results(image=image, regions=regions)

        # return prediction results
        return crop_list, prediction_result, result
