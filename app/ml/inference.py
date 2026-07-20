import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

from app.core.config import get_settings
from app.ml.gradcam import GradCAM, make_overlay
from app.ml.model import CustomCNN
from app.ml.preprocessing import wafer_to_tensor


@dataclass
class InferenceResult:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    is_defective: bool
    inference_ms: float
    heatmap_bytes: bytes
    heatmap_data_uri: str


class ModelService:
    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.class_names = self._load_class_names()
        self.model, self.checkpoint = self._load_model()

    def _load_class_names(self) -> list[str]:
        path = self.settings.label_mapping_path
        if path.exists():
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
            mapping = payload.get("label_to_index", payload)
            return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
        return ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]

    def _load_model(self):
        path = self.settings.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. Copy your best Custom CNN checkpoint into artifacts/."
            )
        checkpoint = torch.load(path, map_location=self.device)
        trial = checkpoint.get("trial", {})
        dropout = float(trial.get("dropout", 0.30))
        number_of_classes = int(checkpoint.get("num_classes", len(self.class_names)))
        if number_of_classes != len(self.class_names):
            raise ValueError("Number of labels does not match model output classes.")
        model = CustomCNN(number_of_classes, dropout)
        state = checkpoint.get("state", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model, checkpoint

    @property
    def model_info(self) -> dict:
        return {
            "architecture": "CustomCNN",
            "device": str(self.device),
            "classes": self.class_names,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "validation_macro_f1": self.checkpoint.get("val_macro_f1"),
        }

    def predict(self, wafer_map) -> InferenceResult:
        input_tensor = wafer_to_tensor(wafer_map).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities_tensor.argmax().item())
        elapsed_ms = (time.perf_counter() - start) * 1000

        gradcam = GradCAM(self.model, self.model.last_conv)
        try:
            cam = gradcam.generate(input_tensor, predicted_index)
        finally:
            gradcam.close()

        overlay = make_overlay(wafer_map, cam)
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Could not encode Grad-CAM heatmap.")
        heatmap_bytes = encoded.tobytes()
        heatmap_b64 = base64.b64encode(heatmap_bytes).decode("ascii")
        probability_values = probabilities_tensor.detach().cpu().tolist()
        probabilities = {
            name: round(float(value) * 100, 4)
            for name, value in zip(self.class_names, probability_values)
        }
        prediction = self.class_names[predicted_index]
        normal_names = {"none", "normal", "no defect", "no_defect"}
        return InferenceResult(
            prediction=prediction,
            confidence=round(float(probability_values[predicted_index]) * 100, 4),
            probabilities=probabilities,
            is_defective=prediction.strip().lower() not in normal_names,
            inference_ms=round(elapsed_ms, 3),
            heatmap_bytes=heatmap_bytes,
            heatmap_data_uri=f"data:image/png;base64,{heatmap_b64}",
        )
