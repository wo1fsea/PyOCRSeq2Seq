from PIL import ImageGrab

from ocr_seq2seq.ocr_seq2seq import OCRSeq2Seq

ocr = OCRSeq2Seq()
last_img = None
while True:
    img = ImageGrab.grabclipboard()
    if img and (not last_img or img.size != last_img.size):
        print(ocr.image_to_string(img))
        last_img = img
