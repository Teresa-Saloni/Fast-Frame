# models/

`fastframe` needs a YOLO-style ONNX model with input `images` `[N,3,640,640]` and output `[N,4+C,A]`.

    pip install ultralytics onnx
    python3 scripts/export_yolo.py                    # -> models/yolov8n.onnx (dynamic batch)
    python3 scripts/quantize_int8.py models/yolov8n.onnx bench/data/calib/   # -> models/yolov8n_int8.onnx

Models are git-ignored; the Docker image copies whatever is in this folder at build time.
