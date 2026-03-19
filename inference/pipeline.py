from core.model_loader import models
from utils.cdr_calculation import calculate_cdr
from utils.vessel_density import vessel_density_score
from inference.feature_builder import build_feature_vector


def run_pipeline(image):

    deep_features = models.resnet.extract_features(image)

    disc_box, cup_box = models.rcnn.detect(image)

    cdr = calculate_cdr(disc_box, cup_box)

    vessel_mask = models.unet.segment(image)

    vessel_risk = vessel_density_score(vessel_mask)

    features = build_feature_vector(deep_features, cdr, vessel_risk)

    prediction = models.xgb.predict(features)

    return {
        "prediction": prediction.tolist(),
        "cdr": float(cdr),
        "vessel_risk": float(vessel_risk)
    }