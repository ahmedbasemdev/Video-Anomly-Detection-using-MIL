from moviepy.editor import VideoFileClip
import math
import numpy as np

# from utils import common

frame_size = 224


def get_frames(my_segment, size=30):
    """
    :keyword
    this function takes a clip and returns a array of frames of this clip
    :param my_segment: clip
    :param size: number of frames you want of sample of this clip
    :return: numpy array of frames with size 3,224,224,size
    """

    ## Segment Level
    frames = []
    for my_frame in my_segment.iter_frames():
        frames.append(my_frame)
    # print(np.array(frames).shape)
    frames = np.array(frames).reshape((3, frame_size, frame_size, -1))
    try:
        random_indices = np.random.choice(frames.shape[0], size=size, replace=False)
        frames = frames[random_indices]

    except:
        pass
    return frames


def split_video(video_path, num_segments, num_frames):
    """
    This function takes path of video and returns and bag of segments of this video
    it returns 32 instance

    :param video_path: the path of video you want to extract instance of
    :param num_segments: number of instance "segments"
    :param num_frames: number of frames of each segments
    :return: segments array with shape ( num_segments , num_frames , width , height)
    """

    bagof_segments = np.zeros((num_segments, 3, num_frames, frame_size, frame_size))
    # Load the video clip
    clip = VideoFileClip(video_path)
    clip = clip.resize((frame_size, frame_size))

    # Get the duration of the video in seconds
    duration = clip.duration

    # Calculate the duration of each segment
    segment_duration = math.ceil(duration / num_segments)

    # Split the video into segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration)
        if start_time > duration:
            break
        my_segment = clip.subclip(start_time, end_time)

        frames = get_frames(my_segment, num_frames)

        if frames.shape[0] != num_frames:
            continue

        bagof_segments[i] = frames
    bagof_segments = bagof_segments / 255
    return bagof_segments
