import joblib
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix


def logistic_model():
    data = pd.read_csv("data/scaled_data.csv")
    encoder_map = joblib.load("joblib/encoders.joblib")
    personality_encoder = encoder_map["personality"]

    feature_data = data[
        [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]
    ]

    target = data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=52
    )

    lr_model = LogisticRegression()
    lr_model.fit(feature_data_train, target_train)

    joblib.dump(lr_model, "models/lr_model.joblib")

    predictions = lr_model.predict(feature_data_test)

    accuracy = accuracy_score(target_test, predictions)
    print("accuracy score:", accuracy, "\n")

    cm = confusion_matrix(target_test, predictions)
    labels = sorted(target_test.unique())

    # This is stupid in the fact that to decode the labels you have to send it as an array that is why there is [0]
    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {personality_encoder.inverse_transform([label])[0]}"
            for label in labels
        ],
        columns=[
            f"Predicted {personality_encoder.inverse_transform([label])[0]}"
            for label in labels
        ],
    )
    print(cm_df, "\n")

    error_rate = 1 - accuracy
    print("Error Rate:", error_rate)

    results = {
        "accuracy_score": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": cm_df.to_dict(),
    }

    with open("scores/lr_scores.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


def random_Forest():
    data = pd.read_csv("data/scaled_data.csv")
    encoder_map = joblib.load("joblib/encoders.joblib")
    personality_encoder = encoder_map["personality"]

    feature_data = data[
        [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]
    ]
    target = data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=52
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(feature_data_train, target_train)

    joblib.dump(rf_model, "models/rf_model.joblib")

    predictions = rf_model.predict(feature_data_test)

    accuracy = accuracy_score(target_test, predictions)
    print("accuracy score:", accuracy, "\n")

    cm = confusion_matrix(target_test, predictions)
    labels = sorted(target_test.unique())

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {personality_encoder.inverse_transform([label])[0]}"
            for label in labels
        ],
        columns=[
            f"Predicted {personality_encoder.inverse_transform([label])[0]}"
            for label in labels
        ],
    )
    print(cm_df, "\n")

    error_rate = 1 - accuracy
    print("Error Rate:", error_rate)

    results = {
        "accuracy_score": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": cm_df.to_dict(),
    }

    with open("scores/rf_scores.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


def neural_Net():
    pass


if __name__ == "__main__":
    logistic_model()
    random_Forest()
    neural_Net()
