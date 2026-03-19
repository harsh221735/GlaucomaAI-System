import torch

model_path = r"C:\Users\Saif\glaucoma-ai-system\models\weights\model.pth"

try:
    model = torch.jit.load(model_path)
    print("Loaded as TorchScript")
except:
    model = torch.load(model_path, map_location="cpu")
    print("Loaded as standard PyTorch model")

torch.save(model, "clean_model.pth")

print("✅ Converted to clean_model.pth")