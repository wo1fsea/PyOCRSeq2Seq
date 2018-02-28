# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/2/11
Description:
    ocr_seq2seq.py
----------------------------------------------------------------------------"""
import numpy as np
from .ocr_model import OCRModel
from .utils import split_text_image, convert_image_to_input_data, convert_input_data_to_image

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 32


class OCRSeq2Seq(object):

    def __init__(self):
        super(OCRSeq2Seq, self).__init__()
        self._ocr_model = OCRModel(IMAGE_WIDTH, IMAGE_HEIGHT)
        self._ocr_model.load_config_for_predict_model(r"D:\GITHUB\TheChineseRoom\ocr_model\checkpoint_256x32_1layer_bGRU")

    def image_to_string(self, image):
        imgs = split_text_image(image, IMAGE_WIDTH / IMAGE_HEIGHT)
        input_data = []
        for line in imgs:
            for img in line:
                input_data.append(convert_image_to_input_data(img, IMAGE_WIDTH, IMAGE_HEIGHT))
                convert_input_data_to_image(input_data[-1]).show()

        size = len(input_data)
        input = np.ones([size, *(input_data[0].shape)])

        for i, data in enumerate(input_data):
            input[i] = data

        strings = self._ocr_model.predict(input)

        result = ""
        i = 0
        for line in imgs:
            for img in line:
                result += strings[i]
                i += 1

            result += "\n"

        return result
