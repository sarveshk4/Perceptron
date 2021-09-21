import os
import joblib

def prepare_data(df):
    """It is used to separate the dependent variable and independent variable

    Args:
        df (pd.DataFrame): its the pandas DataFrame to

    Returns:
        tuple: it return the tuple of dependent and independent variable
    """
    X = df.drop("y",axis=1)
    y=df["y"]
    return X,y

def save_model(model,filename):
    """This saves the trained model 

    Args:
        model (python object): trained model to
        filename (str): path to save the trained model
    """
    model_dirs = "models"
    os.makedirs(model_dirs,exist_ok=True)
    filePath = os.path.join(model_dirs,filename)
    joblib.dump(model,filePath)