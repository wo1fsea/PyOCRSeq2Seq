# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/2/7
Description:
    utils.py
----------------------------------------------------------------------------"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from keras import backend as K
from skimage.filters import threshold_otsu, threshold_yen

FONT_FILE_SUFFIX = [".ttf"]


def text_to_label(text, alphabet):
	return [alphabet.index(char) for char in text]


def label_to_text(labels, alphabet):
	return "".join([alphabet[c] for c in labels])


def ctc_decode(y_preds, greedy=True, beam_width=128):
	labels, probs = K.ctc_decode(y_preds, np.ones((y_preds.shape[0],)) * y_preds.shape[1], greedy, beam_width)
	labels = K.eval(labels[0])
	return labels


def _find_peeks(data, min_val=0, min_range=2):
	start = None
	peeks = []
	for i, val in enumerate(data):
		if val > min_val:
			if start is None:
				start = i
		else:
			if start is not None:
				if i - start >= min_range:
					peeks.append((start, i))
					start = None
	if start is not None:
		peeks.append((start, len(data) - 1))

	return peeks


def convert_image_to_binary(image, self_adapting=True):
	threshold = threshold_otsu(image)
	binary = image > threshold

	if self_adapting:
		if np.sum(binary) > binary.size / 2:
			binary = 1 - binary

	return np.uint8(binary * 255)


def split_char_from_text_image(image):
	image = image.filter(ImageFilter.SHARPEN)

	image_data = np.asarray(image.convert("L"), dtype=np.uint8)
	image_data = convert_image_to_binary(image_data)

	h_sum = np.sum(image_data, axis=1)
	hps = _find_peeks(h_sum)

	boxes = []
	for peek_range in hps:
		start_y = peek_range[0]
		end_y = peek_range[1]
		line_img = image_data[start_y:end_y, :]
		vertical_sum = np.sum(line_img, axis=0)
		vps = _find_peeks(vertical_sum)
		boxes.append([])
		for vp in vps:
			boxes[-1].append((vp[0], start_y, vp[1], end_y))

	images = []
	for bs in boxes:
		images.append([])
		for b in bs:
			images[-1].append(image.crop(b))

	return images


def split_text_image(image, max_ratio=128 / 16):
	image = image.resize((image.size[0] * 2, image.size[1] * 2), Image.NEAREST)

	# image = image.filter(ImageFilter.SHARPEN)

	image_data = np.asarray(image.convert("L"), dtype=np.uint8)
	image_data = convert_image_to_binary(image_data)
	# image = Image.fromarray(image_data)

	h_sum = np.sum(image_data, axis=1)
	hps = _find_peeks(h_sum)

	boxes = []
	for peek_range in hps:
		start_y = peek_range[0]
		end_y = peek_range[1]
		line_img = image_data[start_y:end_y, :]
		vertical_sum = np.sum(line_img, axis=0)
		vps = _find_peeks(vertical_sum)
		boxes.append([])

		start_x = last_x = vps[0][0]
		max_length = (end_y - start_y) * max_ratio

		for vp in vps:
			for x in vp:
				if x - start_x > max_length:
					if last_x != start_x:
						boxes[-1].append((start_x, start_y, last_x, end_y))
						start_x = last_x
					else:
						start_x = x
				last_x = x

		if start_x != last_x:
			boxes[-1].append((start_x, start_y, last_x, end_y))

	images = []
	for bs in boxes:
		images.append([])
		for b in bs:
			images[-1].append(image.crop(b))

	return images


def convert_image_to_input_data(image, image_width, image_height):
	image = image.convert(mode="L")
	image_tmp = Image.new(mode="L", size=(image_width, image_height), color=255)

	# bbox = image.getbbox()
	# image = image.crop(bbox)

	bbox = (0, 0, *image.size)
	if bbox:
		x, y, x2, y2 = bbox
		w, h = x2 - x, y2 - y
		x, y = (0, 0)

		if w / h > image_width / image_height:
			h = round(h / w * image_width)
			w = round(image_width)

		else:
			w = round(w / h * image_height)
			h = round(image_height)

		image = image.resize((w, h), Image.BILINEAR)
		image_tmp.paste(image, box=(x, y, x + w, y + h))

	image_array = np.asarray(image_tmp, dtype=np.uint8)
	# image_array = convert_image_to_binary(image_array, False)
	input_data = convert_image_array_to_input_data(image_array)
	return input_data


def convert_input_data_to_image(input_data):
	image_array = convert_input_data_to_image_array(input_data)
	return Image.fromarray(image_array, mode="L")


def convert_input_data_to_image_array(input_data):
	if K.image_data_format() == "channels_first":
		temp_data = input_data[0, :, :]
	else:
		temp_data = input_data[:, :, 0]

	temp_data = temp_data.T * 255
	temp_data = temp_data.astype(np.uint8)
	return temp_data


def convert_image_array_to_input_data(image_array):
	temp_data = image_array.T
	temp_data = temp_data.astype(np.float32) / 255

	if K.image_data_format() == "channels_first":
		temp_data = np.expand_dims(temp_data, 0)
	else:
		temp_data = np.expand_dims(temp_data, 2)

	return temp_data


def get_input_data_shape(image_width, image_height, channel=1):
	if K.image_data_format() == "channels_first":
		return channel, image_width, image_height
	else:
		return image_width, image_height, channel


def get_font_set_from_dir(dir_path):
	font_set = []
	for root, dirs, files in os.walk(dir_path):
		for f in files:
			file_path = os.path.join(root, f)
			_, ext = os.path.splitext(f)
			if ext.lower() in FONT_FILE_SUFFIX:
				font_set.append(file_path)
	return font_set


def get_random_rgb_color():
	return "#%06X" % random.randrange(0xffffff)
