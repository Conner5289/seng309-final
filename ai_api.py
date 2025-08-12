import use_models
from user_features import UserFeatures
from fastapi import FastAPI

app = FastAPI()


@app.post("/predict")
def predict_(features: UserFeatures):
    use_models.use_logistic_reg()
