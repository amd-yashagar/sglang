# SPDX-License-Identifier: Apache-2.0
"""Experimental gfx950 FP8 MQA: five causal queries reuse each index-K load.

Build hip_gfx950/grouped_mqa.cu with hipcc --offload-arch=gfx950 -O3
-std=c++20 -shared -fPIC -mllvm -amdgpu-mfma-vgpr-form, with GPUs hidden.
Set SGLANG_DSA_INDEXER_HIP_LIB to that library and opt in with
SGLANG_DSA_GFX950_GROUPED_MQA=1. No compilation occurs in this module.
Lengths must be between next_n-1 and width, as in the AITER verify contract.
Only positions below each causal row length are defined.
"""

import ctypes
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def _library(path: str):
    lib = ctypes.CDLL(os.path.realpath(path))
    lib.launch_grouped_mqa.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    )
    lib.launch_grouped_mqa.restype = ctypes.c_int
    return lib


def grouped_mqa_supported(q, cache, weights, lens, tables, width):
    return (
        q.ndim == 4
        and q.shape[1:] == (5, 32, 128)
        and q.dtype == torch.float8_e4m3fn
        and cache.ndim == 4
        and cache.shape[1:] == (64, 1, 132)
        and cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and weights.shape == (q.shape[0] * 5, 32)
        and weights.dtype == torch.float32
        and lens.shape == (q.shape[0],)
        and lens.dtype == torch.int32
        and tables.ndim == 2
        and tables.shape[0] == q.shape[0]
        and tables.dtype == torch.int32
        and width > 0
        and width % 64 == 0
        and tables.shape[1] == width // 64
        and all(
            t.is_cuda and t.device == q.device and t.is_contiguous()
            for t in (q, cache, weights, lens, tables)
        )
        and all(t.data_ptr() % 16 == 0 for t in (q, cache, weights))
    )


def use_grouped_mqa(q, cache, weights, lens, tables, width):
    # Only the measured 32K+ verify regime opts in. Small contexts and decode
    # deliberately preserve the existing dispatch.
    return (
        os.getenv("SGLANG_DSA_GFX950_GROUPED_MQA", "0") == "1"
        and torch.version.hip is not None
        and width >= 32768
        and q.shape[0] >= 16
        and grouped_mqa_supported(q, cache, weights, lens, tables, width)
        and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        == "gfx950"
    )


def gfx950_paged_mqa_logits(q, cache, weights, lens, tables, width, *, out=None):
    if not grouped_mqa_supported(q, cache, weights, lens, tables, width):
        raise ValueError(
            "Grouped MQA requires contiguous gfx950 FP8 Bx5x32x128 queries and page64 cache"
        )
    if (
        torch.version.hip is None
        or torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        != "gfx950"
    ):
        raise ValueError("Grouped MQA requires gfx950")
    shape = (q.shape[0] * 5, width)
    if out is None:
        out = torch.empty(shape, device=q.device, dtype=torch.float32)
    elif (
        out.shape != shape
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid grouped MQA output")
    if not q.shape[0]:
        return out
    path = os.getenv("SGLANG_DSA_INDEXER_HIP_LIB")
    if not path:
        raise RuntimeError(
            "Prebuild grouped_mqa.cu and set SGLANG_DSA_INDEXER_HIP_LIB before enabling grouped MQA"
        )
    # Never launch more wave groups than there are pages. Runtime page counts
    # remain device-side, including under CUDA graph replay.
    splits = min(
        (tables.shape[1] + 3) // 4, max(1, (512 + q.shape[0] - 1) // q.shape[0])
    )
    with torch.cuda.device(q.device):
        code = _library(path).launch_grouped_mqa(
            *[t.data_ptr() for t in (q, cache, weights, lens, tables, out)],
            q.shape[0],
            5,
            tables.shape[1],
            width,
            splits,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if code:
        raise RuntimeError(f"Grouped MQA HIP launch failed: {code}")
    return out
