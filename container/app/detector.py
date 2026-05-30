import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img, new_size):
        w, h = img.size
        scale = new_size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_r = img.resize((new_w, new_h), Image.BILINEAR)
        pad_x = (new_size - new_w) / 2
        pad_y = (new_size - new_h) / 2
        canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
        canvas.paste(img_r, (int(pad_x), int(pad_y)))
        return canvas, scale, (pad_x, pad_y)

    def predict(self, image_path: str) -> list:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = self.session.run(None, {self.input_name: x})[0]  # (1, 300, 6)
        out = out[0]  # (300, 6)

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
            # undo letterbox -> original image pixels
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            # clip to image bounds
            x1 = max(0.0, min(orig_w, float(x1)))
            y1 = max(0.0, min(orig_h, float(y1)))
            x2 = max(0.0, min(orig_w, float(x2)))
            y2 = max(0.0, min(orig_h, float(y2)))
            results.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
