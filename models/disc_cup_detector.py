import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils.preprocessing import preprocess_for_rcnn
from core.config import settings


class DiscCupDetector:

    def __init__(self, model_path):

        checkpoint = torch.load(model_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 🔥 recreate model
        self.model = fasterrcnn_resnet50_fpn(weights=None)

        # ⚠️ adjust if you trained with 3 classes (background, disc, cup)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

        # 🔥 load weights
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def detect(self, image):

        tensor = preprocess_for_rcnn(image)

        with torch.no_grad():
            outputs = self.model(tensor)

        boxes = outputs[0]["boxes"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        disc_box = None
        cup_box = None

        for box, label in zip(boxes, labels):
            if label == 1:
                disc_box = box
            elif label == 2:
                cup_box = box

        if disc_box is None or cup_box is None:
            disc_box = np.array([0, 0, 1, 1])
            cup_box = np.array([0, 0, 1, 1])

        return disc_box, cup_box


def load_rcnn():
    return DiscCupDetector(settings.RCNN_PATH)