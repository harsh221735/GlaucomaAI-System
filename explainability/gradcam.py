import numpy as np

def generate_gradcam(image):

    # placeholder until connected to ResNet layers
    heatmap = np.mean(image, axis=2)

    #return heatmap.tolist()
    return heatmap