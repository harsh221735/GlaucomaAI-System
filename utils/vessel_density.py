import numpy as np

def vessel_density_score(mask):

    vessel_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    density = vessel_pixels / total_pixels

    if density > 0.25:
        return 0
    elif density > 0.15:
        return 0.5
    else:
        return 1