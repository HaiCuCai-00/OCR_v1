import os
import glob
import json
import numpy as np
import cv2
from sys import platform

from .coordinates_process import *
from .text_process import *


def find_anchor_points_and_scale(form_boxes, boxes, texts):
    for key in form_boxes["label"].keys():
        for i, text in enumerate(texts):
            if compare_vietnamese(key, text):
                form_anchor = (
                    np.array(form_boxes["label"][key][0][:2])
                    + np.array(form_boxes["label"][key][0][2:])
                ) / 2.0
                form_w = float(
                    form_boxes["label"][key][0][2] - form_boxes["label"][key][0][0]
                )
                form_h = float(
                    form_boxes["label"][key][0][3] - form_boxes["label"][key][0][1]
                )
                pred_anchor = (np.array(boxes[i][0]) + np.array(boxes[i][2])) / 2.0
                pred_w = float(boxes[i][2][0] - boxes[i][0][0])
                pred_h = float(boxes[i][2][1] - boxes[i][0][1])
                return (
                    form_anchor,
                    pred_anchor,
                    np.array([pred_w / form_w, pred_h / form_h]),
                )
            elif check_start_with_vietnamese(text, key):
                form_anchor = np.array(form_boxes["label"][key][0][:2])
                form_h = float(
                    form_boxes["label"][key][0][3] - form_boxes["label"][key][0][1]
                )
                pred_anchor = np.array(boxes[i][0])
                pred_h = float(boxes[i][2][1] - boxes[i][0][1])
                return (
                    form_anchor,
                    pred_anchor,
                    np.array([pred_h / form_h, pred_h / form_h]),
                )
    return None


def find_form(list_form_paths, boxes, texts):
    best_form_score = 0.0
    out_form_name, out_form_boxes, out_form_anchor, out_pred_anchor, out_scale = (
        None,
        None,
        None,
        None,
        None,
    )
    for form_path in list_form_paths:
        with open(form_path, encoding="utf-8") as f:
            form_boxes = json.load(f)
        form_size = form_boxes["size"]
        anchor_points_and_scale = find_anchor_points_and_scale(form_boxes, boxes, texts)
        if anchor_points_and_scale is None:
            continue
        form_anchor, pred_anchor, scale = anchor_points_and_scale
        form_map = np.zeros((form_size[0], form_size[1]))
        img_map = np.zeros((form_size[0], form_size[1]))
        for key in form_boxes["label"].keys():
            for gt_box in form_boxes["label"][key]:
                for x in range(gt_box[1], gt_box[3]):
                    for y in range(gt_box[0], gt_box[2]):
                        form_map[x, y] = 1
        for key in form_boxes["data"].keys():
            for gt_box in form_boxes["data"][key]:
                for x in range(gt_box[1], gt_box[3]):
                    for y in range(gt_box[0], gt_box[2]):
                        form_map[x, y] = 1
        for box in boxes:
            for x in range(int(box[0][1]), int(box[2][1])):
                for y in range(int(box[0][0]), int(box[2][0])):
                    pred_point = np.array([x, y])
                    pred_point = (pred_point - pred_anchor) / scale[0] + form_anchor
                    try:
                        img_map[int(pred_point[0]), int(pred_point[1])] = 255
                    except:
                        pass

        score = np.sum(form_map * img_map) / np.sum(img_map)
        if score > best_form_score:
            out_form_name = os.path.basename(form_path).split(".")[0]
            best_form_score = score
            out_form_boxes = form_boxes
            out_form_anchor = form_anchor
            out_pred_anchor = pred_anchor
            out_scale = scale

    return out_form_name, out_form_boxes, out_form_anchor, out_pred_anchor, out_scale


def convert_form_with_anchor(
    type, form_boxes, boxes, texts, form_anchor, pred_anchor, scale, form_remove_text
):
    # ----------------------------------- debug -----------------------------------
    form_size = form_boxes["size"]
    form_img = np.zeros((form_size[0] + 200, form_size[1] + 200, 3))
    for key in form_boxes["data"].keys():
        for gt_box in form_boxes["data"][key]:
            for x in range(gt_box[1], gt_box[3]):
                for y in range(gt_box[0], gt_box[2]):
                    form_img[x, y] = [0, 0, 255]
    for i, box in enumerate(boxes):
        for x in range(int(box[0][0]), int(box[2][0]), 20):
            for y in range(int(box[0][1]), int(box[2][1]), 20):
                pred_point = np.array([x, y])
                pred_point = (pred_point - pred_anchor) / scale[0] + form_anchor
                form_img = cv2.circle(
                    form_img,
                    (int(pred_point[0]), int(pred_point[1])),
                    5,
                    (0, 255, 0),
                    -1,
                )
    cv2.imwrite("debug.jpg", form_img)
    # ----------------------------------- debug -----------------------------------

    form_key_and_box_idx = {}
    box_idx_and_form_key = {}
    for key in form_boxes["data"].keys():
        for gt_box in form_boxes["data"][key]:
            for i, box in enumerate(boxes):
                top_left = (np.array(box[0]) - pred_anchor) / scale[0] + form_anchor
                bottom_right = (np.array(box[2]) - pred_anchor) / scale[0] + form_anchor
                pred_box = [
                    int(top_left[0]),
                    int(top_left[1]),
                    int(bottom_right[0]),
                    int(bottom_right[1]),
                ]
                inter_area = get_inter_area(pred_box, gt_box)
                if inter_area > 0:
                    list_box_idx = form_key_and_box_idx.get(key, [])
                    list_box_idx.append(i)
                    form_key_and_box_idx[key] = list_box_idx
                    list_form_key = box_idx_and_form_key.get(i, [])
                    list_form_key.append([key, inter_area])
                    box_idx_and_form_key[i] = list_form_key

    # check if a pred box is belong to two form boxes
    for idx in box_idx_and_form_key.keys():
        if len(box_idx_and_form_key[idx]) > 1:
            max_area = 0
            for (key, area) in box_idx_and_form_key[idx]:
                if max_area < area:
                    max_area = area
            for (key, area) in box_idx_and_form_key[idx]:
                if max_area > area:
                    form_key_and_box_idx[key].remove(idx)

    # get final form
    form = {}
    for key in form_boxes["data"].keys():
        if "form_text" not in key:
            key_value = ""
            for i in form_key_and_box_idx.get(key, []):
                if not check_text_in_list_vietnamese(
                    texts[i], form_boxes["label"].keys()
                ):
                    key_value += " " + texts[i]
            for remove_key in form_remove_text[key]:
                start, end = find_first_similar_substring_idx(key_value, remove_key, string_thresh=80)
                if start is not None and end is not None:
                    key_value = key_value[:start] + " " + key_value[end:]
            form[key] = " ".join(key_value.split())
        else:
            sub_texts = []
            sub_boxes = []
            for i in form_key_and_box_idx.get(key, []):
                sub_texts.append(texts[i])
                sub_boxes.append(boxes[i])
            key_value = merge_text(sub_boxes, sub_texts)
            modules_dir = os.path.dirname(os.path.realpath(__file__))
            with open(
                os.path.join(
                    modules_dir, "form_text_between", f"{type}", f"{key}.json"
                ),
                encoding="utf-8",
            ) as f:
                form_texts = json.load(f)
            sub_form = convert_form_with_text_between(key_value, form_texts)
            for sub_key in sub_form.keys():
                form[sub_key] = sub_form[sub_key]
    return form


def merge_text(boxes, texts):
    merge_text = ""
    lines = find_lines(boxes)
    for line in lines:
        for i in line:
            merge_text += " " + texts[i]
    return " ".join(merge_text.split())


def convert_form_with_text_between(text, form_texts, key_order=True):
    text += " EOF"
    form = {}
    for key in form_texts.keys():
        if "require_" in key:
            continue
        check_require = True
        if f"require_{key}" in form_texts.keys():
            for require_key in form_texts[f"require_{key}"]:
                if require_key not in form.keys():
                    check_require = False
        if not check_require:
            continue
        for (start_key, end_key) in form_texts[key]:
            _, start = find_first_similar_substring_idx(text, start_key)
            end, _ = find_first_similar_substring_idx(text, end_key)
            if start is None or end is None or start >= end:
                continue
            form[key] = " ".join(text[start + 1 : end - 1].split())
            if key_order:
                text = text[end:]
            break
    return form


class FormConverter:
    def __init__(self):
        pass

    def __call__(self, type, boxes, texts):
        modules_dir = os.path.dirname(os.path.realpath(__file__))
        if type in ["dkkd"]:
            text = merge_text(boxes, texts)
            with open(
                os.path.join(modules_dir, "form_text_between", f"{type}.json"),
                encoding="utf-8",
            ) as f:
                form_texts = json.load(f)
            return convert_form_with_text_between(text, form_texts, key_order=False)
        else:
            list_form_paths = glob.glob(
                os.path.join(modules_dir, "form_boxes", f"{type}", "*")
            )
            texts = clean_text_list(texts)

            new_texts = []
            new_boxes = []
            lines = find_lines(boxes)
            for line in lines:
                for i in line:
                    new_texts.append(texts[i])
                    new_boxes.append(boxes[i])
            boxes = new_boxes
            texts = new_texts

            form_name, form_boxes, form_anchor, pred_anchor, scale = find_form(
                list_form_paths, boxes, texts
            )
            if form_name is None:
                return "NOT FOUND"
            with open(
                os.path.join(
                    modules_dir, "form_remove_text", f"{type}", f"{form_name}.json"
                ),
                encoding="utf-8",
            ) as f:
                form_remove_text = json.load(f)
            return convert_form_with_anchor(
                type,
                form_boxes,
                boxes,
                texts,
                form_anchor,
                pred_anchor,
                scale,
                form_remove_text,
            )
