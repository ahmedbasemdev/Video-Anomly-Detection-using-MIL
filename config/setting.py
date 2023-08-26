import json
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    epochs_number: int
    number_frames: int
    number_segments: int
    frame_size: int
    number_videos:int


json_string = '''{

    "epochs_number": 1000,
    "number_frames": 32,
    "number_segments": 32,
    "frame_size": 224,
    "number_videos" : 100
    
}
'''
config_dict = json.loads(json_string)
settings = AppConfig(**config_dict)
