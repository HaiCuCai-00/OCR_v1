from ast import arg
import glob
import os
import sys

from numpy import dtype
from torch import t

sys.path.append("../")
import argparse
import time

import cv2
from omegaconf import OmegaConf
from PIL import Image

from configs import config
from models.craft_text_detector import Craft
from modules.ocr.text_classifier import TextClassifier
from modules.post_processing import FormConverter
from modules.pre_processing import DocScanner, RemoveUnnecessaryPart
from utils.image_processing import ocr_input_processing


class Pipeline:
    def __init__(self):
        self.ocr_config_path = "../configs/ocr/transformer_ocr.yaml"
        self.ocr_cfg = OmegaConf.load(self.ocr_config_path)
        self.init_modules()

    def init_modules(self):
        self.det_model = Craft(crop_type="box", cuda=config.CUDA)
        self.ocr_model = TextClassifier(self.ocr_cfg)
        self.form_converter = FormConverter()
        self.remove_unnecessary_part = RemoveUnnecessaryPart()
        self.scanner = DocScanner()
    def start(self, img, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)
        text_out=''
        try:
            crop_list, regions, image_detect = self.det_model.detect_text(img)
            boxes = regions["boxes"]
            texts = []
            for crop_img in crop_list:
                crop_img = Image.fromarray(crop_img)
                ocr_input = ocr_input_processing(
                    crop_img,
                    self.ocr_cfg.model.input.image_height,
                    self.ocr_cfg.model.input.image_min_width,
                    self.ocr_cfg.model.input.image_max_width,
                )
                text = self.ocr_model.predict(ocr_input)
                print(text)
                texts.append(text)
                text_out += (text +" ") 
            #print(boxes)
            for box in boxes:
                box = box.astype(int)
                cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
            cv2.imwrite("img_with_boxes.png", img)
        except Exception as e:
            text_out=" "
        

        # form = self.form_converter(type, boxes, texts)
        # print(form)
        # return form
        return text_out
    def crop(self, image, x,y,w,h):
        try:
            image =image[y:y+h,x:x+w]
            cv2.imwrite("form.jpg",image)
            return image
        except Exception as e:
            return "box out image"
    def startA(self, img, type, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)
        #img = cv2.resize(img, (1613, 1033))
        # # crop img
        # # img = img[(y):h, (x):w]
        # cv2.imwrite('anh1.jpg',img)
        # img1 =img[457:548,755:1556]
        # cv2.imwrite('anh.jpg',img1)
        crop_list, regions, image_detect = self.det_model.detect_text(img)
        boxes = regions["boxes"]

        texts = []
        for crop_img in crop_list:
            crop_img = Image.fromarray(crop_img)
            ocr_input = ocr_input_processing(
                crop_img,
                self.ocr_cfg.model.input.image_height,
                self.ocr_cfg.model.input.image_min_width,
                self.ocr_cfg.model.input.image_max_width,
            )
            text = self.ocr_model.predict(ocr_input)
            #print(text) 
            texts.append(text)
       # print(texts)      
        for box in boxes:
            box = box.astype(int)
            cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
        cv2.imwrite(os.path.join(output_dir, "img_with_boxes.png"), img)
        form = self.form_converter(type, boxes, texts)
        return form


    def startB(self, img, type, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)

        crop_list, regions, image_detect = self.det_model.detect_text(img)
        boxes = regions["boxes"]

        texts = []
        for crop_img in crop_list:
            crop_img = Image.fromarray(crop_img)
            ocr_input = ocr_input_processing(
                crop_img,
                self.ocr_cfg.model.input.image_height,
                self.ocr_cfg.model.input.image_min_width,
                self.ocr_cfg.model.input.image_max_width,
            )
            text = self.ocr_model.predict(ocr_input)
            texts.append(text)
        #print(type(res))
       # print("text",texts)
        print(" ")
        for box in boxes:
            box = box.astype(int)
            cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
        cv2.imwrite(os.path.join(output_dir, "img_with_boxes.png"), img)
        form = self.form_converter(type, boxes, texts)
        # #if else đến chết để lấy sex
        # if else đến chết để lấy sex
        for i in texts:
            if "Sex" in i: 
                x = i
                S = "Nữ" in x
                if (S == False):
                    form["Giới tính / Sex"] = "Nam"
                ##################
                No = form["Số / No"]
                Nox = No.isdigit()
                if Nox == False:
                    y = No[(len(No)-12):len(No)]
                    form["Số / No"] = y
                ###################
                dob = form["Họ và tên / Full name"]   
                dobx = dob.isupper()
                if dobx == False:
                    x = dob[0:(len(dob)-36)]
                    form["Họ và tên / Full name"] = x
                    z = dob[(len(dob)-10):len(dob)]
                    form["Ngày sinh / Date of birth"] = z
                break
            else:
               x = "Nữ"
        # S = "Nữ" in x
        # if (S == False):
        #     form["Giới tính / Sex"] = "Nam"
        # ##################
        # No = form["Số / No"]
        # Nox = No.isdigit()
        # if Nox == False:
        #     y = No[(len(No)-12):len(No)]
        #     form["Số / No"] = y
        # ###################
        # dob = form["Họ và tên / Full name"]   
        # dobx = dob.isupper()
        # if dobx == False:
        #     x = dob[0:(len(dob)-36)]
        #     form["Họ và tên / Full name"] = x
        #     z = dob[(len(dob)-10):len(dob)]
        #     form["Ngày sinh / Date of birth"] = z
   

        # huy code dạo
        # for i in range(len(texts)):
        #     if "Sex" in texts[i]:
        #         if len(texts[i]) > 18:    
        #             S = texts[i][0:18]
        #             g = texts[i][18:]
        #             texts[i] = S
        #             texts.append(g)
        #         break 
        return form

    def startC(self, img, type, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)

        crop_list, regions, image_detect = self.det_model.detect_text(img)
        boxes = regions["boxes"]

        texts = []
        for crop_img in crop_list:
            crop_img = Image.fromarray(crop_img)
            ocr_input = ocr_input_processing(
                crop_img,
                self.ocr_cfg.model.input.image_height,
                self.ocr_cfg.model.input.image_min_width,
                self.ocr_cfg.model.input.image_max_width,
            )
            text = self.ocr_model.predict(ocr_input)
            print(text) 
            texts.append(text)
        #print(texts)      
        for box in boxes:
            box = box.astype(int)
            cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
        cv2.imwrite(os.path.join(output_dir, "img_with_boxes.png"), img)

        form = self.form_converter(type, boxes, texts)
        if (2>1) or (3<4):
            print("0")
        return form
    def startD(self, img, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)

        crop_list, regions, image_detect = self.det_model.detect_text(img)
        boxes = regions["boxes"]

        texts = []
        for crop_img in crop_list:
            crop_img = Image.fromarray(crop_img)
            ocr_input = ocr_input_processing(
                crop_img,
                self.ocr_cfg.model.input.image_height,
                self.ocr_cfg.model.input.image_min_width,
                self.ocr_cfg.model.input.image_max_width,
            )
            text = self.ocr_model.predict(ocr_input)
            #print(text) 
            texts.append(text)
        for box in boxes:
            box = box.astype(int)
            cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
            cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
        cv2.imwrite(os.path.join(output_dir, "img_with_boxes.png"), img)
        form = texts
        return form
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Document Extraction")
    parser.add_argument("--input", help="Path to single image to be scanned")
    parser.add_argument("--type", help="Document type (cccc/cccd/cmnd/abt/ttk/hc/dkkd/sohong...)")
    parser.add_argument("--output", default="./results", help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pipeline = Pipeline()

    img = cv2.imread(args.input)
    start_time = time.time()
    l = args.type
    print("--------------------------------1")
    if "cccc" in l:
        res = pipeline.startB(img, args.type, args.output)
        print(res)
        print("--------------------------------2")
    else:
        if "cmnd" in l:
            res = pipeline.startA(img, args.type, args.output)
            print(res)
            print("--------------------------------3")
        else:
            if "cccd" in l:
                res = pipeline.startA(img, args.type, args.output)
                print(res)
                print("--------------------------------4")
    if "abt" in l:
        print("--------------------------------5")
        res = pipeline.startD(img, args.output)
        print(res)
        print("--------------------------------5")
    else:
        print("--------------------------------6")
    end_time = time.time()
    print(f"Executed in {end_time - start_time} s")
