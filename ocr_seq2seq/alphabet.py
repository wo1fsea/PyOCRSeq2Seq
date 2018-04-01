# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/2/7
Description:
    alphabet.py
----------------------------------------------------------------------------"""


def alphabet_str2tuple(alphabet):
    alphabet = list(alphabet)
    alphabet.append("")
    return tuple(alphabet)


ALPHABET_KEYBOARD = alphabet_str2tuple("1234567890-="
                                       "!@#$%^&*()_+"
                                       "qwertyuiop[]\\"
                                       "QWERTYUIOP{}|"
                                       "asdfghjkl;'"
                                       "ASDFGHJKL:\""
                                       "zxcvbnm,./"
                                       "ZXCVBNM<>?"
                                       " ")

ALPHABET_LOW_CASE_LETTER = alphabet_str2tuple("abcdefghijklmnopqrstuvwxyz ")
ALPHABET_CAPITAL_LETTER = alphabet_str2tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
ALPHABET_NUM = alphabet_str2tuple("+-1234567890.,")
