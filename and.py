from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_log.log"),level=logging.INFO, format=logging_str,filemode="a")


def main(data, modelName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataFrame {df}")
    X, y = prepare_data(df)
    model = Perceptron(learn_rate=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        main(data=AND, modelName="and_model", eta=ETA, epochs=EPOCHS)
    except Exception as e:
        logging.exception(e)
        raise e