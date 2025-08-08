import onnxruntime as ort
import numpy as np
from pathlib import Path


def load_trained_model():
    path = Path(__file__).parent / "model.onnx"
    return ort.InferenceSession(path)
