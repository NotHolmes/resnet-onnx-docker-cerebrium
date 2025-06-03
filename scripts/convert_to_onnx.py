import torch
from torch import nn
from pathlib import Path

from scripts.pytorch_model import BasicBlock, Classifier  # adjust the import path to where Classifier is defined


class PreprocessingWrapper(nn.Module):
    """Wrapper to preprocess input tensors before passing them to the model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # Normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor before passing it to the model."""
        # Convert to float and scale to [0, 1] (divide by 255)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0  # Divide by 255
        elif x.max() > 1.0:
            x = x / 255.0  # Divide by 255 if not already

        # If input is NHWC, convert to NCHW (RGB format)
        if x.shape[-1] == 3:  # Convert to RGB format if needed
            x = x.permute(0, 3, 1, 2)

        # Normalize using mean and std for each channel (RGB)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)  # Subtract mean and divide std per channel
        return self.model(x)

if __name__ == "__main__":
    # Re-create the trained model according to pytorch_model.py
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    state_dict = torch.load("weights/pytorch_model_weights.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Wrap with preprocessing
    wrapped_model = PreprocessingWrapper(model)
    wrapped_model.eval()

    # Prepare a dummy input tensor
    dummy_input = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)

    output_path = "models/model.onnx"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,  # model to export
        dummy_input,  # example input
        output_path,  # where to save the ONNX file
    )

    print(f"ONNX export complete: {output_path}")
