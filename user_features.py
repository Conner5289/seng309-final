from pydantic import BaseModel


# Need to fix these types, idk what they are just yet
class UserFeatures(BaseModel):
    user_time_alone: float
    user_stage_fear: float
    user_social_event: float
    user_time_outside: float
    user_drained_event: float
    user_friend_num: int
    user_post_num: int
