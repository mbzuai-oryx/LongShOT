# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F



class SoundTower(nn.Module):
    def __init__(self, sound_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.sound_tower_name = sound_tower
        self.cfg_only = None
        
        # Performance optimization caches
        self._attention_mask_cache = {}
        self._seq_range_cache = {}
        self._last_batch_size = None
        self._last_max_seq_len = None

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def _create_attention_mask_optimized(self, audio_feat_lengths, batch_size, max_seq_len, device, dtype):
        """Optimized attention mask creation with caching - torch.compile disabled for compatibility"""
        cache_key = (batch_size, max_seq_len, device, dtype)
        
        # Ensure audio_feat_lengths is on the correct device
        audio_feat_lengths = audio_feat_lengths.to(device=device)
        
        if cache_key not in self._seq_range_cache:
            seq_range = torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=device).unsqueeze(0)
            self._seq_range_cache[cache_key] = seq_range
        else:
            seq_range = self._seq_range_cache[cache_key]
        
        # Expand seq_range safely
        seq_range_expanded = seq_range.expand(batch_size, max_seq_len)
        
        # Ensure proper tensor dimensions and device placement
        if audio_feat_lengths.dim() == 1:
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        else:
            # Handle case where audio_feat_lengths already has extra dimensions
            lengths_expand = audio_feat_lengths.view(batch_size, -1).expand(batch_size, max_seq_len)
        
        # Ensure both tensors are on same device
        lengths_expand = lengths_expand.to(device=device)
        
        padding_mask = seq_range_expanded >= lengths_expand
        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(dtype=dtype, device=device)
        
        # Use proper dtype for -inf assignment to avoid dtype mismatch
        neg_inf_value = torch.tensor(float("-inf"), 
                                           dtype=torch.float, 
                                           device=device)
        audio_attention_mask[audio_attention_mask_] = neg_inf_value
        
        return audio_attention_mask

    def forward(self, sounds, mask=None):
        """Optimized forward pass with caching and reduced computations"""
        
        if isinstance(sounds, list):
            sound_features = []
            for sound in sounds:
                # Calculate attention mask using optimized method
                audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
                batch_size, _, max_mel_seq_len = sound.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                
                # Use optimized attention mask creation
                audio_attention_mask = self._create_attention_mask_optimized(
                    audio_feat_lengths, batch_size, max_seq_len,
                    self.sound_tower.conv1.weight.device,
                    self.sound_tower.conv1.weight.dtype
                )
                
                # Calculate features with mixed precision
                with torch.amp.autocast("cuda", enabled=True):
                    sound_feature = self.sound_tower(sound, attention_mask=audio_attention_mask)
                    sound_feature = sound_feature.to(sound.dtype)
                    sound_feature = sound_feature.last_hidden_state
                sound_features.append(sound_feature)
        else:
            # Optimized batch processing
            if len(sounds.shape) == 5:
                sounds = sounds.squeeze(0).squeeze(1)
                mask = mask.squeeze(0)
                
            # Calculate attention mask using optimized method  
            audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
            
            # Handle variable tensor shapes safely
            if len(sounds.shape) == 4:
                batch_size, _, _, max_mel_seq_len = sounds.shape
            elif len(sounds.shape) == 3:
                batch_size, _, max_mel_seq_len = sounds.shape
            elif len(sounds.shape) == 2:
                batch_size, max_mel_seq_len = sounds.shape
                sounds = sounds.unsqueeze(1)  # Add channel dimension
            else:
                raise ValueError(f"Unexpected sounds tensor shape: {sounds.shape}")
                
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            
            # Use optimized attention mask creation
            audio_attention_mask = self._create_attention_mask_optimized(
                audio_feat_lengths, batch_size, max_seq_len,
                self.sound_tower.conv1.weight.device,
                self.sound_tower.conv1.weight.dtype
            )
            
            # Calculate features with mixed precision
            with torch.amp.autocast("cuda", enabled=True):
                sound_features = self.sound_tower(sounds, attention_mask=audio_attention_mask)
                sound_features = sound_features.last_hidden_state
                sound_features = sound_features.to(sounds.dtype)

        return sound_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.sound_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.sound_tower.config
        else:
            return self.cfg_only
            
    @property
    def device(self):
        return self.sound_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


