import joblib
import pandas as pd
import json
from sklearn.tree import export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

encoder_map = joblib.load("joblib/encoders.joblib")
personality_encoder = encoder_map["personality"]


def logistic_reg():
    data = pd.read_csv("data/scaled_data.csv")

    feature_data = data[
        [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
            "Stage_fear",
            "Drained_after_socializing",
        ]
    ]

    target = data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2
    )

    lr_model = LogisticRegression()
    lr_model.fit(feature_data_train, target_train)

    joblib.dump(lr_model, "models/lr_model.joblib")
    # Get the wights of each columns

    predictions = lr_model.predict(feature_data_test)

    accuracy = accuracy_score(target_test, predictions)

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

    error_rate = 1 - accuracy
    model_wights = lr_model.coef_

    wights_map = {
        "Time_spent_Alone": model_wights[0][0],
        "Social_event_attendance": model_wights[0][1],
        "Going_outside": model_wights[0][2],
        "Friends_circle_size": model_wights[0][3],
        "Post_frequency": model_wights[0][4],
        "Stage_fear": model_wights[0][5],
        "Drained_after_socializing": model_wights[0][6],
    }

    results = {
        "accuracy_score": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": cm_df.to_dict(),
        "wights": wights_map,
    }

    with open("scores/lr_scores.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


def random_Forest():
    data = pd.read_csv("data/scaled_data.csv")

    feature_data = data[
        [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
            "Stage_fear",
            "Drained_after_socializing",
        ]
    ]
    target = data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2
    )

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(feature_data_train, target_train)

    joblib.dump(rf_model, "models/rf_model.joblib")

    predictions = rf_model.predict(feature_data_test)

    accuracy = accuracy_score(target_test, predictions)

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

    error_rate = 1 - accuracy

    results = {
        "accuracy_score": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": cm_df.to_dict(),
    }

    with open("scores/rf_scores.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    feature_names = list(feature_data.columns)

    with open("random_forest_trees.txt", "w") as f:
        for idx, tree in enumerate(rf_model.estimators_):
            f.write(f"\nTree {idx}\n")
            f.write(export_text(tree, feature_names=feature_names))
            f.write("\n" + "=" * 80 + "\n")


def neural_Net():
    pd.options.mode.chained_assignment = None
    data = pd.read_csv("data/scaled_data.csv")

    X = data[
        [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
            "Stage_fear",
            "Drained_after_socializing",
        ]
    ]
    y = data["Personality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Build the model
    nn_model = Sequential()
    nn_model.add(Dense(50, input_dim=X.shape[1], activation="relu"))
    nn_model.add(Dense(100, activation="relu"))
    nn_model.add(Dense(50, activation="relu"))
    nn_model.add(Dense(1, activation="sigmoid"))

    nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    nn_model.fit(X_train, y_train, epochs=50, batch_size=8, shuffle=True, verbose=2)

    # Save the model
    nn_model.save("models/neuralNet.keras")

    # Predict on test set (probabilities)
    probabilities = nn_model.predict(X_test)
    # Convert probabilities to class labels (0 or 1)
    predictions = (probabilities > 0.5).astype(int).flatten()

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    labels = sorted(y_test.unique())

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

    error_rate = 1 - accuracy

    results = {
        "accuracy_score": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": cm_df.to_dict(),
    }

    with open("scores/nn_scores.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    logistic_model()
    random_Forest()
    neural_Net()
