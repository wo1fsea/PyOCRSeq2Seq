from PIL import ImageGrab

from ocr_seq2seq.ocr_seq2seq import OCRSeq2Seq

ocr = OCRSeq2Seq()
last_img = None
while True:
    img = ImageGrab.grabclipboard()
    if img != last_img:
        print(ocr.image_to_string(img))
        last_img = img
