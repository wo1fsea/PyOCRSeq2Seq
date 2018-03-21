# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/1/26
Description:
    data_generator.py
----------------------------------------------------------------------------"""
import numpy as np
from .image_generator import ImageGenerator
from .utils import text_to_label, convert_image_array_to_input_data, get_input_data_shape

MAX_STRING_LEN = 32


class DataGenerator(object):
    def __init__(self, image_width, image_height, output_length, minibatch_size, font_set, alphabet, word_list=[]):
        self.image_generator = ImageGenerator(image_width, image_height, font_set)
        self._image_width = image_width
        self._image_height = image_height
        self._ouput_length = output_length
        self._minibatch_size = minibatch_size
        self._alphabet = alphabet
        self._word_list = word_list

    def _get_random_string(self):
        char = []
        length = np.random.randint(1, MAX_STRING_LEN)
        for i in range(length):
            c = ""
            while not c:
                c = np.random.choice(self._alphabet)
            char.append(c)
        return "".join(char)

    def _get_random_string_avoid_spaces(self):
        string = ""
        while not (len(string) > 0 and any(map(lambda c: c != " ", string))):
            string = self._get_random_string()
        return string

    def _get_input_data(self, text):
        image = self.image_generator.generate(text, rotation=True, translate=False, noise=True)
        return convert_image_array_to_input_data(image)

    def _get_batch_date(self):
        input_data_shape = get_input_data_shape(self._image_width, self._image_height, 1)
        X_data = np.ones([self._minibatch_size, *input_data_shape])

        label = np.ones([self._minibatch_size, MAX_STRING_LEN])
        output_length = np.zeros([self._minibatch_size, 1])
        label_length = np.zeros([self._minibatch_size, 1])

        source_str = []
        for i in range(self._minibatch_size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            text = self._get_random_string_avoid_spaces()
            X_data[i] = self._get_input_data(text)

            t_label = text_to_label(text, self._alphabet)
            label[i, 0:len(t_label)] = t_label
            label_length[i] = len(t_label)
            output_length[i] = self._ouput_length
            source_str.append(text)

        inputs = {
            'image_input': X_data,
            'label': label,
            'output_length': output_length,
            'label_length': label_length,
            'source_str': source_str,
        }
        outputs = {'ctc': np.zeros([self._minibatch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    def get_validate_data(self):
        while 1:
            yield self._get_batch_date()

    def get_train_data(self):
        while 1:
            yield self._get_batch_date()

    def get_test_date(self):
        while 1:
            yield self._get_batch_date()
