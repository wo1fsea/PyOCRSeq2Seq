# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/1/22
Description:
    image_generator.py
----------------------------------------------------------------------------"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .utils import convert_image_to_binary
from keras.preprocessing import image as kpImage
from scipy import ndimage
import math
import random
from keras import backend as K

FONT_SIZE = 32
FONT_SIZE_RANGE = (12, 65)
ROTATION_DEGREE = 5


class ImageGenerator(object):

    def __init__(self, width, height, font_set, font_size_range=FONT_SIZE_RANGE):
        super(ImageGenerator, self).__init__()
        self.width = width
        self.height = height
        self.font_set = font_set
        self.font_size_range = font_size_range

    def add_noise(self, image):
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col)) * 128
        noisy = image + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255
        return noisy.astype(np.uint8)

    def generate(self, string, background=True, rotation=False, translate=False, noise=False):
        """

        :param string:
        :param background:
        :param rotation:
        :param translate:
        :param noise:
        :return:
        """
        font = np.random.choice(self.font_set)
        font_size = np.random.randint(self.font_size_range[0], self.font_size_range[1])

        image = Image.new(mode="RGB", size=(self.width, self.height), color=0xFFFFFF if background else 0)
        image_tmp = Image.new(mode="RGB", size=(font_size * len(string), 2 * font_size))
        draw = ImageDraw.Draw(image_tmp)
        font = ImageFont.truetype(font, font_size)
        draw.text((0, 0), string, font=font)

        if rotation:
            image_tmp = image_tmp.rotate(ROTATION_DEGREE * (np.random.random() - 0.5) * 2, expand=True)

        bbox = image_tmp.getbbox()

        if bbox:
            bbox = bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1
            image_tmp = image_tmp.crop(bbox)

            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y

            if w / h > self.width / self.height:
                h = round(h / w * self.width)
                w = round(self.width)
                x, y = (0, round(np.random.random() * (self.height - h))) if translate else (0, 0)
            else:
                w = round(w / h * self.height)
                h = round(self.height)
                x, y = (round(np.random.random() * (self.width - w)), 0) if translate else (0, 0)

            image_tmp = image_tmp.resize((w, h))  # , Image.LANCZOS)

            image.paste(image_tmp, box=(x, y, x + w, y + h))

        image = image.convert(mode="L")
        image_array = np.asarray(image, dtype=np.uint8)

        if noise:
            image_array = self.add_noise(image_array)

        image_array = convert_image_to_binary(image_array, False)

        return image_array
