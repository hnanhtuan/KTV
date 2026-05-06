import os
import csv
import cv2
import av
import pickle
import torch
import numpy as np
from PIL import Image
import hydra
from tqdm import tqdm
from decord import VideoReader, cpu,gpu
from transformers import Dinov2Model, AutoImageProcessor
from sklearn.cluster import KMeans
import json
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchvision import transforms
fast_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# model_name = "dinov2_vitl14"
# model = torch.hub.load('facebookresearch/dinov2', model_name)
# model.eval()
# model_path = '/home/pengjun/.cache/torch/hub/facebookresearch_dinov2_main'
# # print(2)
# # If CUDA is available, move the model to the GPU.
# device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

model_name_hf = "facebook/dinov2-large"
model = None
hf_processor = None
device = torch.device("cpu")


def select_device(device_name):
    """Select a torch device, falling back to CPU when CUDA is unavailable."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU instead.")
        device_name = "cpu"
    return torch.device(device_name)


def load_dino_model(device_name):
    """Load DINOv2 on the configured device after Hydra config is available."""
    global model, hf_processor, device
    device = select_device(device_name)
    if model is None:
        model = Dinov2Model.from_pretrained(model_name_hf)
        hf_processor = AutoImageProcessor.from_pretrained(model_name_hf)
        model.eval()
    model.to(device)
    print(f"Using DINOv2 device: {device}")


def get_frame_indices(total_frames, max_frames):
    """Return evenly spaced frame indices, capped at max_frames."""
    if total_frames <= max_frames:
        return np.arange(total_frames)
    else:
        return np.linspace(0, total_frames - 1, max_frames, dtype=int)

def get_index( bound, fps, max_frame, first_idx=0):
        """Convert an optional time range into frame indices for JPG frame folders."""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        frame_indices = np.arange(start_idx, end_idx + 1)

        return frame_indices
    
def read_jpg_frame( video_path, bound=None, fps=3):
        """Load JPG frames from a frame folder for an optional time range."""
        print(video_path)
        max_frame = len(os.listdir(video_path))
        frames = list()
        frame_indices = get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        print(frame_indices)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            frames.append(img)

        return frames
    
# def extract_selected_frames(video_path, indices):
#     """Read selected video frames sequentially with OpenCV."""
#     cap = cv2.VideoCapture(video_path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total == 0:
#         return []

#     frames = []
#     indices_set = set(indices)
#     i = 0
#     while i < total:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if i in indices_set:
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(image))
#         i += 1
#     cap.release()
#     return frames


# def extract_selected_frames(video_path, indices):
#     """
#     Use PyAV to efficiently extract frames at the specified indices.
#     :param video_path: video path
#     :param indices: list of frame indices to extract
#     :return: PIL.Image list
#     """
#     container = av.open(video_path)
#     stream = container.streams.video[0]

#     # Force exact decoding.
#     stream.codec_context.skip_frame = "NONE"
#     stream.thread_type = "AUTO"

#     indices_set = set(indices)
#     frames = []
#     current_idx = 0

#     for frame in container.decode(stream):
#         if current_idx in indices_set:
#             img = frame.to_ndarray(format='rgb24')
#             frames.append(Image.fromarray(img))
#         current_idx += 1
#         if current_idx > max(indices):
#             break

#     container.close()
#     return frames

# def extract_selected_frames(video_path, indices):
#     """Batch-read selected frames with Decord and return them as PIL images."""
#     # Create a video reader. Use gpu(0) to enable GPU acceleration.
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total = len(vr)

#     # Filter and sort indices.
#     valid_indices = sorted([i for i in indices if 0 <= i < total])
#     if not valid_indices:
#         return []

#     # Batch-read the specified frames in one call, which is very fast.
#     frames = vr.get_batch(valid_indices).asnumpy()  # shape: (num_frames, H, W, 3)

#     # Convert to a PIL.Image list.
#     images = [Image.fromarray(frame) for frame in frames]
#     return images


def dino_feature_stream(video_path, model, device, indices, batch_size=128):
    """Decode selected video frames and extract pooled DINOv2 features in batches."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    features = []
    current_idx = 0
    indices_set = set(indices)
    max_idx = max(indices)

    with torch.no_grad():
        for frame in container.decode(stream):
            if current_idx in indices_set:
                img = frame.to_ndarray(format='rgb24')
                img = Image.fromarray(img)
                frames.append(fast_transform(img))
                # Extract features and clean up once the batch is full.
                if len(frames) >= batch_size:
                    batch_tensors = torch.stack(frames).to(device)
                    outputs = model(pixel_values=batch_tensors)
                    batch_features = outputs.pooler_output.cpu().numpy()
                    features.extend(batch_features)
                    print('process done')
                    # Clear the cache.
                    del batch_tensors, frames[:]
                    # torch.cuda.empty_cache()
            current_idx += 1
            if current_idx > max_idx:
                break

        # Process remaining frames.
        if frames:
            batch_tensors = torch.stack(frames).to(device)
            outputs = model(pixel_values=batch_tensors)
            batch_features = outputs.pooler_output.cpu().numpy()
            features.extend(batch_features)
            del batch_tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    container.close()
    return features


# def dino_feature_stream_decord(video_path, model, device, indices, batch_size=128):
#     """
#     Use Decord to efficiently extract frames and DINOv2 features.
#     Automatically choose GPU decoding (NVDEC) or CPU decoding.
#     """


#     # Automatically detect whether the GPU is available.
#     # if torch.cuda.is_available() and decord.gpu_enabled():
#     #     ctx = gpu(0)
#     #     print("✅ Using Decord GPU decoding (NVDEC)")
#     # else:
#     ctx = cpu(0)


#     # Read the video.
#     vr = VideoReader(video_path, ctx=ctx)
#     total_frames = len(vr)
#     if total_frames == 0:
#         print(f"⚠️ Empty or unreadable video: {video_path}")
#         return []

#     # Filter valid indices.
#     indices = np.array([i for i in indices if 0 <= i < total_frames])
#     if len(indices) == 0:
#         print(f"⚠️ No valid frame indices for {video_path}")
#         return []

#     features = []

#     # Extract and process frames by batch efficiently.
#     with torch.no_grad():
#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i:i + batch_size]

#             # Batch-read frames with Decord, automatically using GPU or CPU.
#             batch_frames = vr.get_batch(batch_idx).asnumpy()  # (B, H, W, 3)
#             # Convert to PIL and transform.
#             imgs = [fast_transform(Image.fromarray(img)) for img in batch_frames]
#             batch_tensors = torch.stack(imgs).to(device, non_blocking=True)

#             # Extract DINO features.
#             outputs = model(pixel_values=batch_tensors)
#             batch_features = outputs.pooler_output.cpu().numpy()
#             features.extend(batch_features)

#             del batch_tensors, imgs, batch_frames
#             torch.cuda.empty_cache()

#             print(f"Processed {i+len(batch_idx)}/{len(indices)} frames")

#     return features

def dino_feature_optimized(video_path, data_type=None, ts=None,batch_size=256, target_frames=5400):
    """Select frames from a video and run the streaming DINOv2 feature extractor."""
    if data_type=='frame':
        print('frame')
        frames = read_jpg_frame(video_path, ts)
    else:
        # print(video_path)
        # exit(0)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames",total_frames)
        cap.release()

        if total_frames <= 0:
            return []

        indices = get_frame_indices(total_frames, target_frames)
        print(len(indices))
        print(video_path)
        # frames = extract_selected_frames(video_path, indices)
    print(indices)
    # features = []
    features = dino_feature_stream(video_path, model, device, indices, batch_size=512)

    # for i in range(0, len(frames), batch_size):
    #     batch = frames[i:i + batch_size]
    #     # inputs = hf_processor(images=batch, return_tensors="pt").to(device)
    #     inputs = torch.stack([fast_transform(img) for img in batch])
    #     batch_tensors = inputs.to(device, non_blocking=True)  
    #     print('process done')
    #     with torch.no_grad():
    #         outputs = model(pixel_values=batch_tensors)   # <-- Note: use a keyword argument.
    #         batch_features = outputs.pooler_output

    #     # with torch.no_grad():
    #     #     outputs = model(**inputs)
    #     #     batch_features = outputs.pooler_output
    #     features.extend(batch_features.cpu().numpy())

    return features


def dinov2(json_path, video_path, save_tensor_path, dataset):
    """Extract and save DINOv2 frame features for each video listed in a QA JSON file."""
    
    if not os.path.exists(save_tensor_path):
        os.makedirs(save_tensor_path)
    # video_total = set()
    video_total = []
    procesed_videos = set()
    with open(json_path ,'r')as f:
        data = json.load(f)
    for i in data:
        video_total.append(i['question_id'])
    # data = data[250:]
    for video in tqdm(data, total=len(video_total)):
        if dataset=='MLVU_Test':
            video_name = video['video']
        else:
            video_name = video['video_name']
        # if video_name=='test_TFS-5.mp4' or video_name=='test_BWB-5.mp4'or video_name=='test_movie101_91.mp4' or video_name=='test_AWD-3.mp4':
        #     continue
        # full_video_path = video['video_path']
        # if dataset=='MLVU_Test':
        #     folders = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, name))]
        #     # print(folders)
        #     for folder in folders:
        #         if video['question_type'] in folder:
        #             full_video_path = os.path.join(video_path, folder,video_name[5:])
        #             # print(full_video_path)
        #     # exit(0)
        # else:
        #     full_video_path = os.path.join(video_path, video_name)
        full_video_path = os.path.join(video_path, video_name)
        if not os.path.exists(full_video_path):
            print('path not exist',full_video_path)
        video_name = video_name.replace('.mp4','')
        output_path = os.path.join(save_tensor_path, f'{video_name}.pkl')
        # print(save_tensor_path)
        # print(output_path)
        # print(output_path)
        if os.path.exists(output_path):
            continue
        if video_name in procesed_videos:
            continue
        procesed_videos.add(video_name)
        if os.path.exists(output_path):
            continue
        # data_type = video['data_type']
        # print('type',video['data_type'])
        # if video['ts']:
        #     ts = [video['start'], video['end']]
        # else:
        #     ts = None
        # frame_features = dino_feature_optimized(full_video_path, data_type, ts,batch_size=2000)
        frame_features = dino_feature_optimized(full_video_path,batch_size=1000)
        if frame_features:
            temp = {video_name: frame_features}
            with open(output_path, 'wb') as f:
                pickle.dump(temp, f)

def resolve_path(path):
    """Resolve local Hydra config paths relative to the original launch directory."""
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


@hydra.main(config_path="configs/keyframe_select", config_name="config", version_base=None)
def main(cfg: DictConfig):
    load_dino_model(cfg.device)
    dinov2(
        resolve_path(cfg.json_path),
        resolve_path(cfg.video_path),
        resolve_path(cfg.save_tensor_path),
        cfg.dataset,
    )


if __name__ == "__main__":
    main()
