import data_preprocessing
import ai_api
import train_models
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


def test_train_nn():
    train_models.neural_Net()


def test_lr_model():
    user_data = UserFeatures(
        user_time_alone=5.0,
        user_stage_fear="Yes",
        user_social_event=3.0,
        user_time_outside=5.0,
        user_drained_event="Yes",
        user_friend_num=2.0,
        user_post_num=3.0,
    )
    test = ai_api.predict_log(user_data)
    print(test)


def test_rf_model():
    user_data = UserFeatures(
        user_time_alone=5.0,
        user_stage_fear="Yes",
        user_social_event=3.0,
        user_time_outside=5.0,
        user_drained_event="Yes",
        user_friend_num=2.0,
        user_post_num=3.0,
    )
    test = ai_api.predict_forest(user_data)
    print(test)


def test_train_lr_model():
    train_models.logistic_reg()


def test_train_rf_model():
    train_models.random_Forest()


if __name__ == "__main__":
    test_train_nn()
