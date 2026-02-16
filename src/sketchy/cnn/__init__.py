from pathlib import Path

import onnxruntime as ort


def load_trained_model(path=None):
    if path is None:
        path = Path(__file__).parent / "model.onnx"
    return ort.InferenceSession(path)
