import torch

from src.pytorch_model import BasicBlock, Classifier  # adjust the import path to where Classifier is defined

# Re-create the trained model according to pytorch_model.py
model = Classifier(BasicBlock, [2, 2, 2, 2])
state_dict = torch.load("weights/pytorch_model_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Prepare a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,  # model to export
    dummy_input,  # example input
    "models/model.onnx",  # where to save the ONNX file
)

print("ONNX export complete: model.onnx")  # noqa
