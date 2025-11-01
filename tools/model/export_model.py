import argparse
from ultralytics import YOLO
import os

""" Use this script to export the Yolo models to different formats, different precision and for different devices (M1, NVIDIA GPU, CPU).

Note:   For the best performance on M1 use a PyTorch (.pt) model and then run the inference with device='mps'.
        Exporting to ONNX only runs on the CPU for M1 and is much slower.
        A lower precision does not change the performance on M1.

Note:   For the best performance on NVIDIA GPU use a TensorRT (.engine) model and then export the model and run the inference with device='0' (0 for GPU).
        The TensorRT model is optimized for NVIDIA GPU and provides the best performance.
        A lower precision (FP16 or INT8) can improve the inference time on NVIDIA GPU but will reduce the accuracy.
"""

# *****************  Configuration  ************************
YOLO_MODEL_NAME = "yolo11s.pt"  # Pre-trained model file
YOLO_EXPORT_NAME = "yolo11s_16fp_gpu"    # Name without extension
EXPORT_FORMAT = "engine"          # Options: 'onnx', 'engine'
DEVICE='0'                    # 'cpu' for CPU, '0' for GPU, 'mps' for Mac
HALF=True                      # True, use FP16 precision
# ***********************************************************

def export_model(model_name, export_name, export_format, device, half):
    """Export a YOLO model to a different format.

    Args:
        model_name (str): The name of the model file to export.
        export_name (str): The name to use for the exported file (without extension).
        export_format (str): The format to export the model to (e.g., 'onnx', 'engine').
        device (str): The device to use for export (e.g., 'cpu', '0', 'mps').
        half (bool): Whether to use half precision (FP16) for the export.
    """
    model = YOLO(model_name, task='detect')
    model.export(format=export_format, imgsz=640, device=device, half=half)
    
    extension = f".{export_format}"
    exported_file = f"{model_name.rsplit('.', 1)[0]}{extension}"
    if os.path.exists(exported_file):
        os.rename(exported_file, f"{export_name}{extension}")

    print(f"Model exported as {export_name}{extension}.")

if __name__ == "__main__":
    export_model(YOLO_MODEL_NAME, YOLO_EXPORT_NAME, EXPORT_FORMAT, DEVICE, HALF)