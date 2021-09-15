import pandas as pd

from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model

AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df= pd.DataFrame(AND)

X,y = prepare_data(df)

LEARN_RATE = 0.3
EPOCHS = 10

model = Perceptron(learn_rate=LEARN_RATE,epochs=EPOCHS)
model.fit(X,y)

_ = model.total_loss()

save_model(model,filename="and_model")