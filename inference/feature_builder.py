import numpy as np

def build_feature_vector(deep_features, cdr, vessel_risk):

    return np.concatenate([
        deep_features,
        np.array([cdr]),
        np.array([vessel_risk])
    ]).reshape(1, -1)