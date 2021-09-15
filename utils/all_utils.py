import os
import joblib

def prepare_data(df):
    X = df.drop("y",axis=1)
    y=df["y"]
    return X,y

def save_model(model,filename):
    model_dirs = "models"
    os.makedirs(model_dirs,exist_ok=True)
    filePath = os.path.join(model_dirs,filename)
    joblib.dump(model,filePath)