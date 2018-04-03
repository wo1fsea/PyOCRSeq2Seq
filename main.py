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

from PIL import Image

from ocr_seq2seq.ocr_seq2seq import OCRSeq2Seq
img = Image.open("C:/Users/wo1fsea/Desktop/ScreenClip.png")
ocr = OCRSeq2Seq()
print(ocr.image_to_string(img))

# train
# from ocr_seq2seq.ocr_model import OCRModel
# from ocr_seq2seq.utils import get_font_set_from_dir
# W = 256
# H = 32
# ocr_m = OCRModel(W, H, font_set=get_font_set_from_dir("./fonts/"))
# ocr_m.train(0, 10000)
