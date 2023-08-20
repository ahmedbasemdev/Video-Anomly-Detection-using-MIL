import json
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    epochs_number: int
    number_frames: int
    number_segments: int
    frame_size: int


json_string = '''{

    "epochs_number": 1000,
    "number_frames": 32,
    "number_segments": 8,
    "frame_size": 112
    
}
'''
config_dict = json.loads(json_string)
settings = AppConfig(**config_dict)