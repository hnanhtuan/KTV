import os
import cv2
import numpy as np
from PIL import Image

# Suppress noisy FFmpeg/OpenCV decoder logs
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


def load_frame(video_path, num_clips=1, num_frms=4):
    assert num_clips == 1
    frame_names = sorted(os.listdir(video_path))
    total_num_frames = len(frame_names)
    desired_num_frames = min(total_num_frames, num_frms)
    frame_idx = get_seq_frames(total_num_frames, desired_num_frames)

    clip_imgs = []
    original_sizes = []
    for i in frame_idx:
        img = Image.open(os.path.join(video_path, frame_names[i]))
        clip_imgs.append(img)
        original_sizes.append(img.size)
    original_sizes = tuple(original_sizes)
    return clip_imgs, original_sizes


def read_jpg_frame(video_path, keyframe, bound=None, fps=3):
    frame_indices = keyframe
    frames = list()
    original_sizes = []
    for frame_index in frame_indices:
        if isinstance(frame_index, (list, tuple, np.ndarray)):
            frame_idx = frame_index[0] + 1
        else:
            frame_idx = frame_index + 1
        img = Image.open(os.path.join(video_path, f"{frame_idx:05d}.jpg"))
        frames.append(img)
        original_sizes.append(img.size)
    return frames, tuple(original_sizes)


def load_video(video_path, keyframe=None, num_clips=1, num_frms=6, start=None, end=None):
    if os.path.isdir(video_path):
        ts = [start, end]
        return read_jpg_frame(video_path, keyframe, ts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    clip_start = 0 if start is None else int(start * fps)
    clip_end = total_num_frames if end is None else int(end * fps)
    clip_start = max(0, min(clip_start, total_num_frames - 1))
    clip_end = max(clip_start + 1, min(clip_end, total_num_frames))

    assert num_clips == 1

    frame_idx = []
    if keyframe:
        for k in keyframe:
            if isinstance(k, (list, tuple, np.ndarray)):
                frame_idx.append(k[0])
            else:
                frame_idx.append(k)
        frame_idx = sorted(list(set(frame_idx)))
    else:
        n = clip_end - clip_start
        m = min(num_frms, n)
        interval = n / m
        frame_idx = [int(clip_start + i * interval) for i in range(m)]
        frame_idx = sorted(list(set(frame_idx)))

    clip_imgs = []
    original_sizes = []

    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx} from {video_path}")
            continue
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            clip_imgs.append(img)
            original_sizes.append(img.size)
        except Exception as e:
            print("video_path", video_path, "\n", "idx", idx, "\n", e)

    cap.release()
    original_sizes = tuple(original_sizes)
    return clip_imgs, original_sizes


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq
