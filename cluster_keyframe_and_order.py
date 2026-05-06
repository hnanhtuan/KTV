import torch
import clip
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import av
from tqdm import tqdm
import csv
import os
import json
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import os
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
processed_video = dict()

def load_frame(video_path, num_clips=1, num_frms=4):
    # Currently, this function supports only 1 clip
    assert num_clips == 1

    frame_names = sorted(os.listdir(video_path))
    total_num_frames = len(frame_names)

    # Calculate desired number of frames to extract
    desired_num_frames = min(total_num_frames, num_frms)

    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_num_frames, desired_num_frames)

    # Extract frames and get original sizes
    clip_imgs = []
    original_sizes = []
    for i in frame_idx:
        img = Image.open(os.path.join(video_path, frame_names[i]))
        clip_imgs.append(img)
        original_sizes.append(img.size)
    original_sizes = tuple(original_sizes)

    return clip_imgs, original_sizes
def get_index( bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        frame_indices = np.arange(start_idx, end_idx + 1)
        total_frames = len(frame_indices)
        selected_indices_float = np.linspace(0, total_frames - 1, 50)

        selected_indices_int = np.round(selected_indices_float).astype(int)


        selected_frames = frame_indices[selected_indices_int]
        print(selected_frames)
        
        return selected_frames

def read_jpg_frame( video_path, keyframe,bound=None, fps=3):
        print(video_path)
        frame_indices = keyframe
        max_frame = len(os.listdir(video_path))
        frames = list()
        
        original_sizes = []
        for frame_index in frame_indices:
            frame_index+=1
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            print(f"{frame_index:05d}.jpg")
            frames.append(img)
            original_sizes.append(img.size)
        
        return frames,tuple(original_sizes)

def load_video(video_path, keyframe=None, num_clips=1, num_frms=6, start=None, end=None):
    """
    Load video frames from a video file.

    Parameters:
    - video_path (str): Path to the video file.
    - keyframe (list): List of keyframe tuples like [(10,), (20,), ...] (optional).
    - num_clips (int): Number of clips to extract. Only 1 is supported.
    - num_frms (int): Number of frames to extract.
    - start (float or None): Start time in seconds. None means from beginning.
    - end (float or None): End time in seconds. None means till end.

    Returns:
    - clip_imgs (list[PIL.Image.Image]): List of extracted frames.
    - original_sizes (tuple): Tuple of frame sizes.
    """
    if os.path.isdir(video_path):
        ts = [start, end]
        frames,original_sizes = read_jpg_frame(video_path, keyframe,ts)
        return frames,original_sizes
        # print(frame_idx)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('total_num_frames',total_num_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_num_frames / fps

        # Convert start/end time (in seconds) to frame indices
        clip_start = 0 if start is None else int(start * fps)
        clip_end = total_num_frames if end is None else int(end * fps)

        clip_start = max(0, min(clip_start, total_num_frames - 1))
        clip_end = max(clip_start + 1, min(clip_end, total_num_frames))
        # print('clip_start',clip_start,clip_end)
        if clip_end <= clip_start:
            raise ValueError(f"Invalid start/end seconds: start={start}s, end={end}s")

        # Compute frame indices
        print('keyframe',keyframe)
        if keyframe:
            # frame_idx = sorted([k[0] for k in keyframe if clip_start <= k[0] < clip_end])
            # frame_idx = sorted([k for k in keyframe if clip_start <= k < clip_end])
            frame_idx = sorted([k for k in keyframe])
        else:
            n = clip_end - clip_start
            m = min(num_frms, n)
            interval = n / m
            frame_idx = [int(clip_start + i * interval) for i in range(m)]
    print('frame_idx',frame_idx)
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
            print('video_path', video_path, '\n', 'idx', idx, '\n', e)

    cap.release()
    original_sizes = tuple(original_sizes)
    print('len',len(clip_imgs))
    return clip_imgs, original_sizes


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq







def extract_selected_frames(video_path, indices):
    extracted_frames_pil = []
    indices_set = set(indices)
    max_needed_index = -1
    if indices_set: 
        max_needed_index = max(indices_set)

    try:
        with av.open(video_path) as container:
            try:
                stream = container.streams.video[0]
            except IndexError:
                return []
            if stream.frames == 0:
                frame_idx_counter = 0
                frames_found_count = 0
            for frame in container.decode(stream):
                if frame_idx_counter in indices_set:
                    pil_image = frame.to_image() 
                    extracted_frames_pil.append(pil_image)
                    frames_found_count += 1
            
                if frames_found_count == len(indices_set):
                    break 

                if max_needed_index != -1 and frame_idx_counter >= max_needed_index:
                    break
                
                frame_idx_counter += 1
            
            if frames_found_count < len(indices_set):
                actual_video_frames = frame_idx_counter 



    except FileNotFoundError:
        return []
    except av.AVError as e:

        return []
    except Exception as e: 

        return []
            
    return extracted_frames_pil

def video_frame_clustering(frame_features: np.ndarray, num_cluster: int=5):
    
    """Cluster video frames using K-means and return the indices of the cluster centers."""
    kmeans = KMeans(n_clusters=num_cluster, random_state=0, init='k-means++', n_init=10).fit(frame_features)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    distances = np.linalg.norm(frame_features - cluster_centers[:, np.newaxis, :], axis=2)
    closest_frames = np.argmin(distances, axis=1)
    clusters = [[] for _ in range(num_cluster)]
    cluster_center_indices = [[] for _ in range(num_cluster)] 
    for j, label in enumerate(labels):
        clusters[label].append(j)
    for j in range (num_cluster):
        cluster_center_indices[j] = closest_frames[j]
    for j in range(len(cluster_center_indices)):
        cluster_center_indices[j] = int(cluster_center_indices[j])
    return cluster_center_indices


def get_original_frame_number(
    total_original_frames: int,
    index_in_extracted_list: int,  
    ts: list = None, 
    fps: int = None,  
    max_frames_to_extract: int = 5400
) -> int:  
    print(total_original_frames)

    num_actually_extracted: int
    if total_original_frames <= max_frames_to_extract:
        num_actually_extracted = total_original_frames
    else:
        num_actually_extracted = max_frames_to_extract

    if index_in_extracted_list >= num_actually_extracted:
        raise ValueError(
            f"index_in_extracted_list ({index_in_extracted_list}) out range"
        )

    original_frame_index_0_based: int

    if total_original_frames <= max_frames_to_extract:

        if ts:
            original_frame_index_0_based = int(ts[0]*fps)+ index_in_extracted_list
        else:
            original_frame_index_0_based =  index_in_extracted_list
    else:

        if num_actually_extracted == 1:

            if index_in_extracted_list == 0:
                original_frame_index_0_based = 0
            else:
 
                raise ValueError("If only one frame is extracted, index_in_extracted_list must be 0.")
        else:

            # original_frame_index_0_based = round(
            #     index_in_extracted_list * (total_original_frames - 1) / (num_actually_extracted - 1)
            # )
            # original_frame_index_0_based = int(original_frame_index_0_based) 
            changed_list = np.linspace(0, total_original_frames - 1, max_frames_to_extract, dtype=int)
            # print(index_in_extracted_list)
            original_frame_index_0_based = changed_list[index_in_extracted_list]
            # print(original_frame_index_0_based)
      
    return original_frame_index_0_based


def cluster(json_path, video_path, video_frame_tensor_path, save_cluster_path, dataset):
    """Select question-aware keyframes and save their ranked order per question."""
    # This function first uses precomputed DINOv2 frame features to find diverse
    # candidate frames, then uses CLIP to rank those candidates against the question.
    num_keyframes = 12
    max_frames_to_extract = 5400

    # Create the output folder that will hold one JSON file per question.
    os.makedirs(save_cluster_path, exist_ok=True)

    # Load the precomputed DINOv2 features produced by keyframe_select_new.py.
    # Expected shape per video: one feature vector for each sampled frame.
    with open(video_frame_tensor_path, 'rb') as f:
        video_frame_tensor = dict(pickle.load(f))

    # Load the QA metadata. Each item tells us which video and question to process.
    with open(json_path, 'r') as f:
        qa_data = json.load(f)

    for sample in tqdm(qa_data, total=len(qa_data)):
        # Pull the fields needed to locate the video, rank frames for the question,
        # and name the per-question output JSON.
        video_name = sample['video_name']
        prompt = sample['question']
        question_id = sample['question_id']

        # Skip samples that were already processed in a previous run.
        output_path = os.path.join(save_cluster_path, f'{question_id}.json')
        if os.path.exists(output_path):
            continue

        # Read video metadata so indices from the sampled DINO feature list can be
        # mapped back to original video frame numbers.
        full_video_path = os.path.join(video_path, video_name)
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Fetch this video's DINOv2 feature matrix. If it is missing, this question
        # cannot be clustered, so skip it.
        tensor = video_frame_tensor.get(video_name)
        if tensor is None:
            print('error', question_id)
            continue

        # KMeans selects diverse candidate frames by finding the frame nearest to
        # each cluster center in DINOv2 feature space.
        print(len(tensor))
        clustered_indices = video_frame_clustering(tensor, num_keyframes)
        print('cluster_frame_temp', clustered_indices)

        # DINO features may have been extracted from at most max_frames_to_extract
        # sampled frames. Convert those sampled-list indices back to original frame IDs.
        candidate_frame_indices = [
            get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for index in clustered_indices
        ]

        # If mapping creates duplicate original frame IDs, fall back to evenly spaced
        # frames so the downstream CLIP ranking still receives num_keyframes candidates.
        if len(set(candidate_frame_indices)) != num_keyframes:
            candidate_frame_indices = [
                int(index)
                for index in np.linspace(0, float(total_frames), num_keyframes, endpoint=False)
            ]

        print('candidate_frame_indices', candidate_frame_indices)
        # Load the actual image frames for the candidate original frame indices.
        frames_cluster, _ = load_video(
            full_video_path,
            candidate_frame_indices,
            start=None,
            end=None,
        )

        # Convert candidate frames into CLIP input tensors.
        image_batch = [preprocess_clip(frame) for frame in frames_cluster]

        # CLIP has a limited text context. For long questions, keep a middle slice
        # that likely contains the most useful content words.
        prompt_words = prompt.split(' ')
        print('prompt_len', len(prompt_words))
        if len(prompt_words) > 40:
            prompt = ' '.join(prompt_words[10:50])

        # Encode candidate frames and the question with CLIP, normalize both feature
        # sets, then compute image-text similarity scores.
        image_input = torch.stack(image_batch).to(device)
        with torch.no_grad():
            print('question', prompt)
            image_features = model_clip.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(device)
            text_features = model_clip.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T

        print(similarity)
        # Rank candidates by CLIP similarity. The highest scoring frame gets rank 0.
        top_k_count = min(num_keyframes, similarity.numel())
        _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)
        print(top_k_indices)

        # Convert ranked candidate positions back to original video frame numbers.
        top_keyframes = [
            candidate_frame_indices[int(index)]
            for index in top_k_indices
        ]

        # Store each selected frame with its rank. In inference, this order is used
        # to allocate more visual tokens to more question-relevant keyframes.
        key_frame_order = [
            [frame_index, rank]
            for rank, frame_index in enumerate(top_keyframes)
        ]

        # Save one JSON file per question: {question_id: [[frame_index, rank], ...]}.
        result = {question_id: key_frame_order}
        print(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: int(o) if isinstance(o, np.integer) else o,
            )


json_path_videomme_test = 'ktv/playground/gt_qa_files/Videomme/val_qa.json'
save_tensor_path_videomme_test = 'ktv/save_tensor/Videomme.pkl'
video_path_videomme = 'datasets/Video-MME/data'
save_cluster_path_videomme_test = 'videomme_test_json_temp12'
cluster(json_path_videomme_test, video_path_videomme,save_tensor_path_videomme_test, save_cluster_path_videomme_test,'videomme')
