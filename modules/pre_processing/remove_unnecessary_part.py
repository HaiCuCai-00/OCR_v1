import cv2
import numpy as np


def get_mean_color(img):
    color = np.mean(np.mean(img, axis=0), axis=0)
    return color


class RemoveUnnecessaryPart:
    def __init__(self):
        pass

    def __call__(self, type, img):
        h, w, c = img.shape
        if type == "ttk":
            blue_color = np.array([150, 90, 80])
            for i in range(0, h, 200):
                for j in range(0, w, 200):
                    gid_color = get_mean_color(img[i : i + 200, j : j + 200, :])
                    if np.sum(np.abs(blue_color - gid_color)) < 70:
                        img[i : i + 200, j : j + 200, :] = blue_color
        return img
