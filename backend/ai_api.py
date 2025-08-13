import use_models
import train_models
import json
from data_preprocessing import preprocess_user_data
from user_features import UserFeatures
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static HTML
@app.get("/")
def get_frontend():
    return FileResponse("../frontend/index.html")


@app.post("/predict_lr")
def predict_log(features: UserFeatures):
    user_df = preprocess_user_data(features)
    return use_models.use_logistic_reg(user_df)


@app.post("/predict_rf")
def predict_forest(features: UserFeatures):
    user_df = preprocess_user_data(features)
    return use_models.use_random_forest(user_df)


@app.post("/predict_nn")
def predict_neural(features: UserFeatures):
    user_df = preprocess_user_data(features)
    return use_models.use_neuarl_network(user_df)


@app.post("/train_models")
def train_all():
    train_models.logistic_reg()
    train_models.random_Forest()
    train_models.neural_Net()

    with open("scores/lr_scores.json") as f:
        logistic_status = json.load(f)
    with open("scores/rf_scores.json") as f:
        random_forest_status = json.load(f)
    with open("scores/nn_scores.json") as f:
        neural_net_status = json.load(f)

    combined_status = {
        "logistic_regression": logistic_status,
        "random_forest": random_forest_status,
        "neural_network": neural_net_status,
    }
    return combined_status
