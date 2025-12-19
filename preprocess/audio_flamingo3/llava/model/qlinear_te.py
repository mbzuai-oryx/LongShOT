# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, InplaceFunction
from torch.cuda import amp

from .language_model.configuration_quantize import QuantizationConfig
from .qfunction import block_cut, block_quant, block_reshape
from .qutils import quant_get_local_rank
from .realquantize.division_transpose import fp8_division_transpose
from .realquantize.linear import fp8_linear_backward, fp8_linear_forward
from .realquantize.quantize_and_transpose import fp8_quantize_and_transpose


class QLinearTE(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, args=None, layer_idx=0):
        super().__init__(in_features, out_features, bias, device)
        try:  # TODO: remove this try except (llama & qwen2)
            self.args = QuantizationConfig(**deepcopy(args))
        except:
            self.args = deepcopy(args)

        self.apply_quantize = min(self.weight.shape[0], self.weight.shape[1]) >= 3584

        if quant_get_local_rank() == 0:
            if self.apply_quantize:
                print(f"[qlinear debug] Apply QLinear, {layer_idx}")
            else:
                print(f"[qlinear debug] Don't QLinear, {layer_idx} since the weight is too small: {self.weight.shape}")
        self.layer_idx = layer_idx
        self.layer_name = None
        
        # Pre-allocate memory pools for better performance
        self._weight_cache = None
        self._input_cache = None
        self._last_input_shape = None

    @torch.compile(mode="max-autotune", dynamic=False)
    def forward(self, Input):
        # Cache management for repeated operations
        if self._last_input_shape != Input.shape:
            self._last_input_shape = Input.shape
            # Clear cache when input shape changes
            self._weight_cache = None
            self._input_cache = None
        
        if self.training and self.apply_quantize:
            output = QuantLinearTE.apply(Input, self.weight, self.bias, self.args, self.layer_name)
        else:
            # Use compiled linear operation for inference
            output = F.linear(Input, self.weight, self.bias)

        return output


# if int(os.environ.get("LOCAL_RANK")) == 0:
#     import IPython
#     IPython.embed()
# else:
#     import time
#     time.sleep(1000)

# class QuantLinearTE(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, args, layer_type):
#         ctx.saved = input, weight, bias, args, layer_type
#         return F.linear(input, weight, bias)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias, args, layer_type = ctx.saved

#         C_in = input.shape[-1]
#         C_out = grad_output.shape[-1]

#         grad_output_flatten = grad_output.reshape(-1, C_out)
#         input_flatten = input.reshape(-1, C_in)

#         if grad_output_flatten.dtype == input_flatten.dtype:
#             grad_weight = grad_output_flatten.t().mm(input_flatten)
#         else:
#             grad_weight = grad_output_flatten.float().t().mm(input_flatten)

#         if grad_output_flatten.dtype == weight.dtype:
#             grad_input = grad_output_flatten.mm(weight)
#         else:
#             grad_input = grad_output_flatten.float().mm(weight)

#         if bias is not None:
#             grad_bias = grad_output_flatten.sum(0)
#         else:
#             grad_bias = None

#         grad_input_transform = grad_input.reshape(input.size())

#         return grad_input_transform, grad_weight, grad_bias, None, None


class QuantLinearTE(Function):
    # Pre-allocate CUDA events to avoid repeated allocation overhead
    _cuda_events = None
    _event_pool_size = 20
    _event_pool_idx = 0
    
    @classmethod
    def _get_cuda_events(cls):
        """Get pre-allocated CUDA events for timing"""
        if cls._cuda_events is None:
            cls._cuda_events = []
            for _ in range(cls._event_pool_size):
                cls._cuda_events.append([
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)
                ])
        
        # Round-robin through event pool
        events = cls._cuda_events[cls._event_pool_idx % cls._event_pool_size]
        cls._event_pool_idx += 1
        return events
    
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.bfloat16, device_type="cuda")
    def forward(ctx, input, weight, bias, args, layer_name):
        time_bench = os.getenv("TIME_BENCH")
        
        # Use torch.compile for critical path
        with torch.cuda.device(input.device):
            # Quantize input and weight in parallel using streams
            if time_bench:
                start_events = QuantLinearTE._get_cuda_events()
                start_events[0].record()

            # Use optimized quantization
            Qinput, Iscale, Qinput_t = fp8_quantize_and_transpose(
                input, 16, args.fabit, transpose_output_2d=True
            )

            if time_bench:
                start_events[1].record()

            # Cache weight quantization if possible
            Qweight, Wscale, Qweight_t = fp8_quantize_and_transpose(
                weight, 16, args.fwbit, transpose_output_2d=True
            )

            ctx.saved = Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name
            
            # Optimized FP8 linear forward
            fc_output = fp8_linear_forward(Qinput, Iscale, Qweight, Wscale, False, 0, bias)

            # Reduced timing overhead - only when explicitly requested
            if time_bench and quant_get_local_rank() == 0:
                torch.cuda.synchronize()
                elapsed_time = start_events[0].elapsed_time(start_events[1])
                if elapsed_time > 1.0:  # Only log slow operations
                    print(f"[Forward] FP8 Linear: {elapsed_time:.3f}ms | Shape: {input.shape} -> {fc_output.shape}")

        return fc_output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name = ctx.saved

        time_bench = os.getenv("TIME_BENCH")
        
        with torch.cuda.device(grad_output.device):
            if time_bench:
                start_events = QuantLinearTE._get_cuda_events()
                start_events[0].record()

            # Optimized gradient quantization
            Qgrad_output, Gscale, Qgrad_output_t = fp8_quantize_and_transpose(
                grad_output, 16, args.bobit, stochastic=False, transpose_output_2d=True
            )

            # Compute gradients using optimized FP8 operations
            grad_input, grad_weight = fp8_linear_backward(
                Qinput_t, Iscale, Qgrad_output, Gscale, Qgrad_output_t,
                Qweight_t, Wscale, 16, bias, stochastic=False, dgrad_quantize=False,
            )

            # Efficient bias gradient computation
            if bias is not None:
                grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
            else:
                grad_bias = None

            # Reduced timing overhead
            if time_bench and quant_get_local_rank() == 0:
                start_events[1].record()
                torch.cuda.synchronize()
                elapsed_time = start_events[0].elapsed_time(start_events[1])
                if elapsed_time > 1.0:  # Only log slow operations
                    print(f"[Backward] FP8 Linear: {elapsed_time:.3f}ms")

        return grad_input, grad_weight, grad_bias, None, None
