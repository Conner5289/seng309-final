import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error


def linear_model():
    data = pd.read_csv("data/scaled_data.csv")
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

    lr_model = LinearRegression()
    lr_model.fit(feature_data_train, target_train)
    mae_test = mean_absolute_error(target_test, lr_model.predict(feature_data_test))
    print("The error rate for this model is ", mae_test)
    joblib.dump(lr_model, "models/lr_model.joblib")


def decision_tree():
    feature_data = encoded_data[
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
    target = encoded_data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=52
    )

    dt_model = DecisionTreeClassifier(random_state=52)
    dt_model.fit(feature_data_train, target_train)
    mae_test = mean_absolute_error(target_test, dt_model.predict(feature_data_test))
    print("The error rate for this model is", mae_test)
    return dt_model


def neural_Net():
    pass


if __name__ == "__main__":
    linear_model()
    decision_tree()
    neural_Net()
