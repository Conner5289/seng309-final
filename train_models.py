import joblib
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score


def logistic_reg():
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
pd.options.mode.chained_assignment = None

# Show all columns when printing dataframes
pd.set_option('display.max_columns', None)

def train_model():
    data = pd.read_csv("data/scaled_data.csv")
   # x
    X = data[[
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]]
    # y variable (0 = introvert, 1 = extrovert)
    y = data["Personality"]

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Build the model
    model = Sequential()
    model.add(Dense(50, input_dim=7, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, shuffle=True, verbose=2)
    # Save the trained model using the specified name and format
    model.save('neuralNet.keras')
    print("Training complete and model saved as 'neuralNet.keras'")
 

if __name__ == "__main__":
    logistic_model()
    random_Forest()
    neural_Net()
