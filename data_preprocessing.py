from numpy._core.multiarray import scalar
from numpy.random.mtrand import f
import pandas as pd
from pandas.io.sql import com
from sklearn import preprocessing
from sklearn.externals.array_api_compat.numpy import e
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import joblib
from user_features import UserFeatures


def preprocess_data():
    # Reads the csv file of data that trains our model:
    raw_data = pd.read_csv("personality_datasert.csv")

    # delete any rows with empty data before any preprocessing
    raw_data.dropna(inplace=True)

    # Makes encoder, encodes the data from a dict then kicks the dict to a new data frame with everything
    label_encoder_stage_fear = preprocessing.LabelEncoder()
    label_encoder_drained = preprocessing.LabelEncoder()
    label_encoder_personality = preprocessing.LabelEncoder()

    encoded_data_dict = {
        "Stage_fear": label_encoder_stage_fear.fit_transform(raw_data["Stage_fear"]),
        "Drained_after_socializing": label_encoder_drained.fit_transform(
            raw_data["Drained_after_socializing"]
        ),
        "Personality": label_encoder_personality.fit_transform(raw_data["Personality"]),
    }

    encoded_dataframe = pd.DataFrame(encoded_data_dict)

    encoder_map = {
        "stage_fear": label_encoder_stage_fear,
        "drained": label_encoder_drained,
        "personality": label_encoder_personality,
    }
    joblib.dump(encoder_map, "joblib/encoders.joblib")

    # Feature Scaling, scaled every feature that wasn't label encoded:
    data_scaler = MinMaxScaler()

    features_to_scale = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    unscaled_dict = {
        "Time_spent_Alone": raw_data["Time_spent_Alone"],
        "Social_event_attendance": raw_data["Social_event_attendance"],
        "Going_outside": raw_data["Going_outside"],
        "Friends_circle_size": raw_data["Friends_circle_size"],
        "Post_frequency": raw_data["Post_frequency"],
    }

    unscaled_df = pd.DataFrame(unscaled_dict, columns=features_to_scale)

    scaled_array = data_scaler.fit_transform(unscaled_df)
    joblib.dump(data_scaler, "joblib/data_scaler.joblib")

    scaled_dataFrame = pd.DataFrame(scaled_array, columns=features_to_scale)
    scaled_dataFrame.reset_index(drop=True, inplace=True)

    final_dataframe = pd.concat([scaled_dataFrame, encoded_dataframe], axis=1)
    final_dataframe.reset_index(drop=True, inplace=True)
    final_dataframe.to_csv("data/scaled_data.csv")


def preprocess_user_data(features: UserFeatures):
    encoders = joblib.load("joblib/encoders.joblib")

    label_encoder_stage_fear = encoders["stage_fear"]
    label_encoder_drained = encoders["drained"]

    encoded_dict = {
        "Stage_fear": label_encoder_stage_fear.transform([features.user_stage_fear]),
        "Drained_after_socializing": label_encoder_drained.transform(
            [features.user_drained_event]
        ),
    }
    encoded_df = pd.DataFrame(encoded_dict)

    data_scaler = joblib.load("joblib/data_scaler.joblib")

    features_to_scale = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    unscaled_dict = {
        "Time_spent_Alone": [features.user_time_alone],
        "Social_event_attendance": [features.user_social_event],
        "Going_outside": [features.user_time_outside],
        "Friends_circle_size": [features.user_friend_num],
        "Post_frequency": [features.user_post_num],
    }

    unscaled_df = pd.DataFrame(unscaled_dict)[features_to_scale]
    scaled_array = data_scaler.transform(unscaled_df)
    scaled_df = pd.DataFrame(scaled_array, columns=features_to_scale)
    scaled_df.reset_index(drop=True, inplace=True)

    final_df = pd.concat([scaled_df, encoded_df], axis=1)
    return final_df


if __name__ == "__main__":
    preprocess_data()
