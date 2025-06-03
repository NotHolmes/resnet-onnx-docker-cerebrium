import numpy as np
import onnxruntime as ort
from PIL import Image

from src.class_labels import CLASS_LABELS

IMG_SIZE = 224
IMG_CHANNELS = 3
ALPHA_CHANNELS = 4


class OnnxModel:
    """ONNX model wrapper for loading and making predictions from PIL images."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        """Preprocess a PIL image for ONNX model prediction."""
        # Resize to 224x224 using bilinear interpolation (default)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        # Convert to numpy array and float32
        img = np.array(img).astype(np.float32)
        # If grayscale, stack to get 3 channels (RGB)
        if img.ndim == IMG_CHANNELS - 1:
            img = np.stack([img] * IMG_CHANNELS, axis=-1)
        # If image has alpha channel, discard it
        if img.shape[2] == ALPHA_CHANNELS:
            img = img[:, :, :IMG_CHANNELS]
        # Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        # This is required for compatibility with ONNX models
        img = img.transpose(2, 0, 1)  # HWC to CHW
        # Scale pixel values to [0, 1]
        img /= 255.0
        # Normalize using mean and std for each channel (RGB)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img - mean) / std  # Subtract mean and divide by std per channel
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def predict(self, img: Image.Image) -> np.ndarray:
        """Run prediction on a PIL image and return the model output.

        The ONNX model expects a raw RGB uint8 image of shape (1, 224, 224, 3)
        because preprocessing is inside the model. We still convert to RGB and
        resize here since ONNX can't process PIL images or handle color/size conversion.
        """
        # Ensure image is RGB and resized
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img, dtype=np.uint8)

        # Add batch dimension
        input_tensor = np.expand_dims(img_np, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]

    def predict_index(self, img: Image.Image) -> int:
        """Predict the class index for a given image."""
        output = self.predict(img)
        return int(np.argmax(output, axis=1)[0])


if __name__ == "__main__":
    img_path = "images/n01667114_mud_turtle.JPEG"
    model_path = "models/model.onnx"

    img = Image.open(img_path)
    model = OnnxModel(model_path)
    output = model.predict_index(img)
    print(f"Predicted: {CLASS_LABELS[output]} (index {output})")
