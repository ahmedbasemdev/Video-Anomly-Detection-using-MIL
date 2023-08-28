import json
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    epochs_number: int = 1000
    number_frames: int = 30
    number_segments: int = 32
    frame_size: int = 224
    number_videos:int = 100


# json_string = '''{
#
#     "epochs_number": 1000,
#     "number_frames": 32,
#     "number_segments": 32,
#     "frame_size": 224,
#     "number_videos" : 100
#
# }
# '''
# config_dict = json.loads(json_string)
# settings = AppConfig(**config_dict)
