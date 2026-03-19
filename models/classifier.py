import xgboost as xgb

def load_xgboost():

    model = xgb.Booster()
    model.load_model("models/weights/xgboost_model.pkl")

    return model