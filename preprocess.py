"""
Preprocessing utilities for deepfake detection pipeline.
Includes face detection and cropping using OpenCV Haar Cascade.
"""

import cv2
import numpy as np
from PIL import Image


# Load OpenCV face detector
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(image: Image.Image, padding: float = 0.2) -> Image.Image:
    """
    Detect the largest face in the image and return a padded crop.
    Falls back to the full image if no face is detected.

    Args:
        image:   PIL RGB image
        padding: Fractional padding around the face bounding box (default 20%)
    Returns:
        PIL RGB image (cropped to face, or original if no face found)
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return image  # No face found – use full image

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Apply padding
    ph, pw = int(h * padding), int(w * padding)
    H, W = image.height, image.width
    x1 = max(0, x - pw)
    y1 = max(0, y - ph)
    x2 = min(W, x + w + pw)
    y2 = min(H, y + h + ph)

    return image.crop((x1, y1, x2, y2))


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 numpy array (H, W, C)."""
    return np.array(image)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL image."""
    return Image.fromarray(arr.astype(np.uint8))


def resize(image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
    return image.resize(size, Image.LANCZOS)
