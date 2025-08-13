import joblib
from user_features import UserFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
import numpy as np

encoders = joblib.load("joblib/encoders.joblib")
personality_encoder = encoders["personality"]


def use_logistic_reg(user_df):
    print("Using lr")
    lr_model = joblib.load("models/lr_model.joblib")
    lr_prediction = lr_model.predict(user_df)
    return personality_encoder.inverse_transform(lr_prediction)[0]


def use_random_forest(user_df):
    print("Using rf")
    rf_model = joblib.load("models/rf_model.joblib")
    rf_prediction = rf_model.predict(user_df)
    return personality_encoder.inverse_transform(rf_prediction)[0]


def use_neuarl_network(user_df):
    print("Using NN")
    nn_model = load_model("models/neuralNet.keras")
    pred_prob = nn_model.predict(user_df)
    nn_prediction = (pred_prob > 0.5).astype(int).flatten()
    personality = personality_encoder.inverse_transform(nn_prediction)[0]
    return personality
