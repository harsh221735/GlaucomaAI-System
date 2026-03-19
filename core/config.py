class Settings:

    MODEL_DIR = "models/weights"

    RESNET_PATH = r"C:\Users\Saif\glaucoma-ai-system\models\weights\resnet_feature_extractor.pth"
    RCNN_PATH = r"C:\Users\Saif\glaucoma-ai-system\models\weights\rcnn_detector.pth"
    UNET_PATH = r"C:\Users\Saif\glaucoma-ai-system\models\weights\unet_vessel_segmentation.pth"
    XGB_PATH = r"C:\Users\Saif\glaucoma-ai-system\models\weights\xgboost_model.pkl"

    PATIENT_DB = "data/patient_records.json"

    IMAGE_SIZE = 224

    LLM_MODEL = "llama3"


settings = Settings()