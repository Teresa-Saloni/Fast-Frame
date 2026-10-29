#!/usr/bin/env python3
"""Export YOLOv8n to ONNX with a dynamic batch dim (what ort_detector.cpp expects).
   pip install ultralytics onnx
   python3 scripts/export_yolo.py          -> models/yolov8n.onnx
"""
import pathlib
from ultralytics import YOLO
out = pathlib.Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
m = YOLO("yolov8n.pt")
path = m.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=17)
pathlib.Path(path).rename(out / "yolov8n.onnx")
print("wrote", out / "yolov8n.onnx")
