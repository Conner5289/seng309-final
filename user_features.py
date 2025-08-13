from pydantic import BaseModel


class UserFeatures(BaseModel):
    user_time_alone: float
    user_stage_fear: str
    user_social_event: float
    user_time_outside: float
    user_drained_event: str
    user_friend_num: float
    user_post_num: float
