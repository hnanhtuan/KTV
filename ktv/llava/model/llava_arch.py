#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# --------------------------------------------------
# Based on the code by Haotian liu, 2020
# Under the Apache License, Version 2.0
# --------------------------------------------------
from abc import ABC, abstractmethod
import math
import re

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape

from einops import rearrange


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(
        self,
        images,
        keyframe_order=None,
        num_frames=None,
        prune_mode=None,
        rate=None,
        tokens_num=None,
    ):
        """
        Encode image/video frame patches and optionally prune patch tokens.

        Inputs:
        - images: pixel tensor consumed by the vision tower.
        - prune_mode: None for no token pruning, or one of
          {"cls_new_token_sim", "uniform_token"} for score-based pruning.
        - keyframe_order/num_frames/tokens_num/rate: pruning controls.

        Returns:
        - Encoded (projected) patch features. Shape follows existing behavior:
          - no pruning: [B, N, D] (or [1, T*N, D] for frame inputs)
          - pruning:    [1, K_total, D]
        """
        model = self.get_model()
        vision_tower = model.get_vision_tower()
        projector = model.mm_projector

        supported_pruned_modes = {"cls_new_token_sim", "uniform_token"}
        if prune_mode is not None and prune_mode not in supported_pruned_modes:
            raise ValueError(f"Unsupported prune_mode: {prune_mode}")

        # Fast path: no pruning, simply drop CLS token then project.
        if prune_mode is None:
            image_features = vision_tower(images, prune_mode)  # [B, N+1, C]
            image_features = image_features[:, 1:]  # remove CLS -> [B, N, C]
            image_features = projector(image_features)  # -> [B, N, D]
            if num_frames is not None:
                # Keep legacy frame layout used by current downstream code.
                image_features = image_features.flatten(0, 1).unsqueeze(0)
            return image_features

        # Pruning path: vision tower also returns CLS-to-token importance scores.
        image_features, cls_att = vision_tower(images, prune_mode)
        selected_projected_features = []

        for frame_idx in range(len(image_features)):
            # Remove CLS token and keep only patch tokens for this frame.
            frame_patch_features = image_features[frame_idx][1:]

            # Decide how many tokens to keep for this frame.
            if keyframe_order:
                order = keyframe_order[frame_idx]
                if num_frames == 6:
                    if order == 0:
                        remain_tokens_num = 288
                    elif order in [1, 2, 3, 4]:
                        remain_tokens_num = 144
                    else:
                        remain_tokens_num = 72

                    # Keep original special-case behavior for specific budgets.
                    if tokens_num == 1872:
                        if order in [0]:
                            remain_tokens_num = 576
                        elif order in [1, 2, 3, 4]:
                            remain_tokens_num = 288
                        else:
                            remain_tokens_num = 144
                    if tokens_num == 504:
                        if order in [0]:
                            remain_tokens_num = 144
                        else:
                            remain_tokens_num = 72

                elif num_frames == 12:
                    if order in [0, 1]:
                        remain_tokens_num = 288
                    elif order in [2, 3, 4, 5, 6, 7, 8, 9]:
                        remain_tokens_num = 144
                    else:
                        remain_tokens_num = 72
            else:
                remain_tokens_num = tokens_num // num_frames

            # Build per-token score:
            # 1) cls_att gives token importance wrt CLS.
            # 2) token-token similarity gives redundancy (higher = more redundant).
            # 3) combine them with "rate" as weighting factor.
            cls_sim = cls_att[frame_idx]
            cls_score = (cls_sim - cls_sim.min()) / (cls_sim.max() - cls_sim.min())

            normalized_features = frame_patch_features / frame_patch_features.norm(
                dim=1, keepdim=True
            )
            pairwise_sim = torch.matmul(normalized_features, normalized_features.t())
            pairwise_sim = pairwise_sim.fill_diagonal_(0)
            token_redundancy = torch.sum(pairwise_sim, dim=1) / 575
            redundancy_score = (token_redundancy - token_redundancy.min()) / (
                token_redundancy.max() - token_redundancy.min()
            )

            rate_tensor = torch.tensor(rate, device=cls_score.device)
            combined_score = cls_score * rate_tensor + (1 - redundancy_score) * (
                1 - rate_tensor
            )

            # Keep top-K tokens, then restore token order for stable sequence layout.
            _, top_k_indices = torch.topk(
                combined_score, remain_tokens_num, largest=True
            )
            top_k_indices = top_k_indices.sort()[0]
            selected_tokens = frame_patch_features[top_k_indices]

            # Project selected tokens to language hidden size.
            selected_projected_features.append(projector(selected_tokens).unsqueeze(0))

        image_features = torch.cat(selected_projected_features, dim=1)
        print(image_features.shape)
        return image_features

    def temporal_aggregation(self, image_features, temporal_aggregation):
        """
        Aggregate frame-level patch features across space/time.

        Expected input shape:
        - image_features: [T, N, D]
          T: number of frames
          N: number of patch tokens per frame
          D: feature dimension

        Output shape is always batched as [1, *, D] for downstream consumption.
        """
        T, N, D = image_features.shape

        if temporal_aggregation == "concat":
            # Flatten frame and patch axes: [T, N, D] -> [T*N, D].
            aggregated = image_features.view(T * N, D)

        elif temporal_aggregation == "spatial_1d_max_pool":
            # 1D pool over token axis inside each frame, then concatenate frames.
            pooled = rearrange(image_features, "t n d -> t d n")
            pooled = nn.MaxPool1d(kernel_size=2, stride=2)(pooled)
            pooled = rearrange(pooled, "t d n -> t n d", t=T)
            aggregated = pooled.reshape(-1, D)

        elif temporal_aggregation == "spatial_1d_avg_pool":
            # Same as above but with average pooling.
            pooled = rearrange(image_features, "t n d -> t d n")
            pooled = nn.AvgPool1d(kernel_size=2, stride=2)(pooled)
            pooled = rearrange(pooled, "t d n -> t n d", t=T)
            aggregated = pooled.view(-1, D)

        elif temporal_aggregation == "spatial_2d_max_pool":
            # Rebuild patch grid, apply 2D max pooling per frame, then flatten back.
            n0 = n1 = int(math.sqrt(N))
            pooled = rearrange(
                image_features, "t (n0 n1) d -> d t n0 n1", n0=n0, n1=n1
            )
            pooled = nn.MaxPool2d(kernel_size=2, stride=2)(pooled)
            aggregated = rearrange(pooled, "d t n0 n1 -> (t n0 n1) d")

        elif temporal_aggregation == "spatial_2d_avg_pool":
            # Rebuild patch grid, apply 2D average pooling per frame, then flatten back.
            n0 = n1 = int(math.sqrt(N))
            pooled = rearrange(
                image_features, "t (n0 n1) d -> d t n0 n1", n0=n0, n1=n1
            )
            pooled = nn.AvgPool2d(kernel_size=2, stride=2)(pooled)
            aggregated = rearrange(pooled, "d t n0 n1 -> (t n0 n1) d")

        elif temporal_aggregation == "spatial_temporal_pool":
            # Joint spatio-temporal adaptive pooling to a fixed output volume.
            pooling_size = (16, 12, 12)
            n0 = n1 = int(math.sqrt(N))
            pooled = rearrange(
                image_features, "t (n0 n1) d -> d t n0 n1", n0=n0, n1=n1
            )
            pooled = nn.AdaptiveAvgPool3d(pooling_size)(pooled)
            aggregated = rearrange(pooled, "d t n0 n1 -> (t n0 n1) d")

        elif temporal_aggregation == "temporal_global_pool":
            # Average across frames only: [T, N, D] -> [N, D].
            aggregated = torch.mean(image_features, dim=0)

        else:
            raise ValueError(
                f"Unknown temporal aggregation method: {temporal_aggregation}"
            )

        return aggregated.unsqueeze(0)

    def prepare_ktv(self, image_features, temporal_aggregation):
        """
        Build KTV dual-path features from per-frame patch tokens.

        `temporal_aggregation` format:
        "ktv-slow_{num}frms_{slow_agg}-fast_{h}x{w}"
        Example:
        "ktv-slow_10frms_spatial_1d_max_pool-fast_4x4"

        Output:
        - Concatenation of slow-path and fast-path features, shape [1, K, D].
        """
        T, N, D = image_features.shape

        # Parse KTV config string.
        ktv_match = re.match(
            r"^ktv-slow_(\d+)frms_(\w+)-fast_(\d+)x(\d+)$", temporal_aggregation
        )
        if not ktv_match:
            raise ValueError(
                f"Failed to parse the temporal aggregation for ktv: {temporal_aggregation}"
            )

        num_slowpath_frames = int(ktv_match.group(1))
        slowpath_aggregation = ktv_match.group(2)
        fastpath_output_size = (int(ktv_match.group(3)), int(ktv_match.group(4)))

        # Slow pathway:
        # sample frames uniformly along the timeline, then apply configured aggregation.
        slowpath_idx = torch.linspace(0, T, num_slowpath_frames + 1)
        slowpath_idx = slowpath_idx.to(torch.int32).tolist()
        slowpath_idx.pop()  # remove endpoint T (out of valid index range)
        slowpath_features = self.temporal_aggregation(
            image_features[slowpath_idx], slowpath_aggregation
        )

        # Fast pathway:
        # keep full temporal resolution, but spatially downsample each frame grid.
        n0 = n1 = int(math.sqrt(N))
        fastpath_features = rearrange(
            image_features, "t (n0 n1) d -> d t n0 n1", n0=n0, n1=n1
        )
        fastpath_features = nn.AdaptiveAvgPool2d(fastpath_output_size)(
            fastpath_features
        )
        fastpath_features = rearrange(fastpath_features, "d t n0 n1 -> (t n0 n1) d")
        fastpath_features = fastpath_features.unsqueeze(0)

        # Final KTV token stream: slow-path tokens first, then fast-path tokens.
        return torch.cat((slowpath_features, fastpath_features), dim=1)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
        temporal_aggregation=None,
        keyframe_order=None,
        num_frames=None,
        prune_mode=None,
        global_rate=None,
        tokens_num=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            # image_features = self.encode_images(images)
            import time

            start_time = time.time()
            image_features = self.encode_images(
                images, keyframe_order, num_frames, prune_mode, global_rate, tokens_num
            )
            end_time = time.time()

            print(f"Execution time: {end_time - start_time:.6f} seconds")
            # exit(0)
        # if temporal_aggregation and \
        #    temporal_aggregation.lower() != 'none' and \
        #    temporal_aggregation.lower() != 'false':
        #     if temporal_aggregation.startswith('ktv'):
        #         image_features = self.prepare_ktv(image_features, temporal_aggregation)
        #     else:
        #         image_features = self.temporal_aggregation(image_features, temporal_aggregation)
        # print(image_features.shape)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
