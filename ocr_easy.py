# ocr_easy.py
import re
import cv2
import easyocr

class PlateOCR:
    """
    EasyOCR wrapper for license plates.
    Recognition-only on YOLO crops.
    """
    def __init__(self, lang: str = 'en'):
        # Init EasyOCR once (CPU-friendly)
        langs = [lang] if isinstance(lang, str) else lang
        self.reader = easyocr.Reader(langs)  # downloads models on first run

    @staticmethod
    def preprocess(crop_bgr):
        # Light cleanup helps small, noisy plates
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 7, 75, 75)
        g = cv2.equalizeHist(g)
        # Upscale a bit to help the recognizer
        g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return g

    @staticmethod
    def clean(text: str) -> str:
        # Keep uppercase alphanumerics (typical plates)
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def read(self, crop_bgr):
        """
        Returns (best_text, best_conf).
        EasyOCR returns list of (bbox, text, conf).
        """
        proc = self.preprocess(crop_bgr)
        out = self.reader.readtext(proc, detail=1)  # [(bbox, text, conf), ...]
        if not out:
            return "", 0.0
        # pick most confident
        bbox, txt, conf = max(out, key=lambda x: x[2])
        return self.clean(txt), float(conf or 0.0)
