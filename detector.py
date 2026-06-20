"""
Deepfake Detector using EfficientNet-B0 (pretrained on ImageNet,
fine-tuned for real/fake binary classification).

Swap the model weights path once you have trained weights.
For a quick demo, it uses random weights — replace with your .pth file.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class DeepfakeDetector:
    MODEL_NAME = "efficientnet_b0_deepfake"

    def __init__(self, weights_path: str = None, threshold: float = 0.5):
        self.threshold = threshold
        self.model_name = self.MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(weights_path)
        self.model.eval()

        # ImageNet-style normalization (matches EfficientNet training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _build_model(self, weights_path):
        """Build EfficientNet-B0 with binary output head."""
        model = models.efficientnet_b0(pretrained=(weights_path is None))
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1)   # 1 output -> sigmoid -> probability of FAKE
        )
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state)
            print(f"[Detector] Loaded weights from {weights_path}")
        else:
            print("[Detector] WARNING: Using untrained weights. Predictions are random.")

        return model.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        Args:
            image: PIL Image (RGB)
        Returns:
            {
                "label": "FAKE" | "REAL",
                "is_fake": bool,
                "confidence": float (0-100),
                "fake_probability": float (0-1),
                "real_probability": float (0-1)
            }
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
        logit = self.model(tensor)                                      # (1, 1)
        fake_prob = torch.sigmoid(logit).item()
        real_prob = 1.0 - fake_prob

        is_fake = fake_prob >= self.threshold
        label = "FAKE" if is_fake else "REAL"
        confidence = fake_prob if is_fake else real_prob

        return {
            "label": label,
            "is_fake": bool(is_fake),
            "confidence": round(confidence * 100, 2),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
        }
