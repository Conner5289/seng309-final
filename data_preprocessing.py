import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import joblib


def data_preprocessing():
    # Reads the csv file of data that trains our model:
    raw_data = pd.read_csv("personality_datasert.csv")

    # delete any rows with empty data before any preprocessing
    raw_data.dropna(inplace=True)

    # Makes encoder, encodes the data from a dict then kicks the dict to a new data frame with everything
    label_encoder_stage_fear = preprocessing.LabelEncoder()
    label_encoder_drained = preprocessing.LabelEncoder()
    label_encoder_personality = preprocessing.LabelEncoder()

    encoded_data_dict = {
        "Time_spent_Alone": raw_data["Time_spent_Alone"],
        "Stage_fear": label_encoder_stage_fear.fit_transform(raw_data["Stage_fear"]),
        "Social_event_attendance": raw_data["Social_event_attendance"],
        "Going_outside": raw_data["Going_outside"],
        "Drained_after_socializing": label_encoder_drained.fit_transform(
            raw_data["Drained_after_socializing"]
        ),
        "Friends_circle_size": raw_data["Friends_circle_size"],
        "Post_frequency": raw_data["Friends_circle_size"],
        "Personality": label_encoder_personality.fit_transform(raw_data["Personality"]),
    }

    encoded_dataframe = pd.DataFrame(encoded_data_dict)

    encoder_list = {
        "Stage_fear": label_encoder_stage_fear,
        "Drained": label_encoder_drained,
        "personality": label_encoder_personality,
    }
    joblib.dump(encoder_list, "joblib/encoders.joblib")

    # Makes a csv file with the encoded data
    encoded_dataframe.to_csv("data/encoded_Data.csv", index=False)

    # Feature Scaling, scaled every feature that wasn't label encoded:
    data_scaler = MinMaxScaler()

    features_to_scale = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    norma_array = data_scaler.fit_transform(encoded_dataframe[features_to_scale])

    joblib.dump(data_scaler, "joblib/data_scaler.joblib")

    norma_dataFrame = pd.DataFrame(norma_array, columns=features_to_scale)
    norma_dataFrame.reset_index(drop=True, inplace=True)

    non_norma_columns = encoded_dataframe[
        ["Stage_fear", "Drained_after_socializing", "Personality"]
    ].reset_index(drop=True)

    final_dataframe = pd.concat([norma_dataFrame, non_norma_columns], axis=1)
    final_dataframe.to_csv("data/scaled_data.csv", index=False)


if __name__ == "__main__":
    data_preprocessing()
