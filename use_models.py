import joblib
from user_features import UserFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def use_logistic_reg(features: UserFeatures):
    lr_model = joblib.load("models/lr_model.joblib")


def use_random_forest():
    dt_model = RandomForestClassifier()

    dt_model_prediction = dt_model.predict(user_dataframe_encoded)
    predicted = personality_encoder.inverse_transform(dt_model_prediction)
    print("You are an", predicted[0])


def use_neuarl_network():
    pass
