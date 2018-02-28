# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/2/28
Description:
    main.py
----------------------------------------------------------------------------"""

import numpy as np
from PIL import Image

from ocr_seq2seq.ocr_model import OCRModel
from ocr_seq2seq.utils import split_text_image, convert_image_to_input_data

W = 256
H = 32
ocr_m = OCRModel(W, H)

# predict
# ocr_m.load_config_for_predict_model(r"D:\GITHUB\TheChineseRoom\ocr_model\checkpoint_27")
# img = Image.open("C:/Users/wo1fsea/Desktop/ScreenClip.png")
#
# imgs = split_text_image(img, W/H)
# input_data = []
# for line in imgs:
#     for img in line:
#         input_data.append(convert_image_to_input_data(img, W, H))
#         img.show()
#
# size = len(input_data)
# input = np.ones([size, *(input_data[0].shape)])
#
# for i, data in enumerate(input_data):
#     input[i] = data
#
# print(ocr_m.predict(input, size))

# train
ocr_m.train(0, 1000)