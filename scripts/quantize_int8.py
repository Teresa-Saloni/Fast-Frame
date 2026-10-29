#!/usr/bin/env python3
"""Static INT8 quantization of the exported YOLO model with ONNX Runtime.
   pip install onnxruntime onnx numpy opencv-python
   python3 scripts/quantize_int8.py models/yolov8n.onnx bench/data/calib/   -> models/yolov8n_int8.onnx
Calibration images: ~200 frames from the benchmark video (ffmpeg -i x.mp4 -vf fps=1 calib/%04d.jpg).
"""
import pathlib, sys
import numpy as np, cv2
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

src = pathlib.Path(sys.argv[1]); calib = pathlib.Path(sys.argv[2])
dst = src.with_name(src.stem + "_int8.onnx")

def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s / w, s / h); nw, nh = int(w * r), int(h * r)
    out = np.full((s, s, 3), 114, np.uint8); out[(s - nh) // 2:(s - nh) // 2 + nh, (s - nw) // 2:(s - nw) // 2 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return (out[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[None]

class Reader(CalibrationDataReader):
    def __init__(self): self.files = iter(sorted(calib.glob("*.jpg"))[:200])
    def get_next(self):
        f = next(self.files, None)
        return None if f is None else {"images": letterbox(cv2.imread(str(f)))}

quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
print("wrote", dst)
