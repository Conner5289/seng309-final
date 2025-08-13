import data_preprocessing
from user_features import UserFeatures


def test_data_preprocess():
    data_preprocessing.preprocess_data()


def test_user_data_prec():
    user_data = UserFeatures(
        user_time_alone=5.0,
        user_stage_fear="Yes",
        user_social_event=3.0,
        user_time_outside=5.0,
        user_drained_event="Yes",
        user_friend_num=2.0,
        user_post_num=3.0,
    )

    test = data_preprocessing.preprocess_user_data(user_data)
    print(test)


if __name__ == "__main__":
    test_data_preprocess()
    test_user_data_prec()
