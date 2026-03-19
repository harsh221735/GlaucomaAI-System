import torch
import numpy as np
from utils.preprocessing import preprocess_for_unet
from core.config import settings


# ⚠️ YOU MUST match your training architecture
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # ❗ Replace this with your actual UNet definition
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VesselSegmenter:

    def __init__(self, model_path):

        checkpoint = torch.load(model_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 🔥 recreate model
        self.model = UNet()

        # 🔥 load weights
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def segment(self, image):

        tensor = preprocess_for_unet(image)

        with torch.no_grad():
            output = self.model(tensor)

        mask = torch.sigmoid(output)
        mask = mask.squeeze().cpu().numpy()

        binary_mask = (mask > 0.5).astype(np.uint8)

        return binary_mask


def load_unet():
    return VesselSegmenter(settings.UNET_PATH)