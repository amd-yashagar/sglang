"""Measured, opt-in TP8 K3 BF16 GEMM paths; other linears are unchanged."""

import functools
import json
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def _paths():
    return json.loads(Path(__file__).with_name("mla_gemm.json").read_text())


def mla_gemm_covered(x, weight):
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.ndim == weight.ndim == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and x.stride(0) % 8 == weight.stride(0) % 8 == 0
        and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and f"{x.shape[0]}:{weight.shape[0]}:{weight.shape[1]}" in _paths()
    )


def _flydsl_no_split(x, weight, out, config):
    # The public wrapper allocates/zeros split-K counters even for split_k=1.
    # This specialization never reads them: pass empty typed placeholders,
    # avoiding two Fill kernels captured into every graph. Keep split-K calls
    # on the public wrapper and its capture-scoped semaphore protocol.
    from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as gm

    kwargs = dict(
        config,
        a_is_transposed=False,
        b_is_transposed=True,
        in_dtype_id=gm.GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=gm.GEMM_A16W16_DTYPE_BF16,
        has_bias=False,
    )
    param = gm.make_gemm_a16w16_param_and_validate(
        x.shape[0], weight.shape[0], x.shape[1], kwargs
    )
    assert param is not None and config["split_k"] == 1
    a_arg = gm._dynamic_tensor_arg(x, 1)
    args = (
        gm._dynamic_tensor_arg(out, 1),
        a_arg,
        gm._dynamic_tensor_arg(weight.t(), 0),
        a_arg,
        x.new_empty((0,), dtype=torch.int64),
        x.new_empty((0,), dtype=torch.int32),
        1,
        param,
        torch.cuda.current_stream(x.device),
    )
    gm.run_cached(
        gm.gemm_a16w16_gfx950,
        *args,
        constexpr_param=param,
        compiler=gm.flyc.compile,
        dispatch_args=args,
    )


def prepare_o_proj_weight(weight):
    """One-time/reload preshuffle; retain dense fallback and graph-stable storage.

    This is deliberately called at post-load, never lazily during a forward.
    Keeping the original parameter also preserves checkpoint/loaders and M not
    covered by the measured dispatch. Costs 22 MB per TP8 MLA layer.
    """
    if (
        weight.shape != (7168, 1536)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        return
    from aiter.ops.shuffle import shuffle_weight

    packed = shuffle_weight(weight, layout=(16, 16))
    previous = getattr(weight, "_k31_preshuffled", None)
    if (
        previous is not None
        and previous.shape == packed.shape
        and previous.device == packed.device
    ):
        previous.copy_(packed)
    else:
        weight._k31_preshuffled = packed


def mla_linear_into(x, weight, out):
    if not (
        mla_gemm_covered(x, weight)
        and out.dtype == x.dtype
        and out.device == x.device
        and out.is_contiguous()
        and out.shape == (x.shape[0], weight.shape[0])
    ):
        return False
    cfg = _paths()[f"{x.shape[0]}:{weight.shape[0]}:{weight.shape[1]}"]
    if cfg["kind"] == "hipblaslt":
        torch.mm(x, weight.t(), out=out)
    elif cfg["kind"] == "skinny":
        from aiter.ops.flydsl.kernels.gemm_a16w16_skinny_gfx950 import (
            gemm_a16w16_skinny,
        )

        gemm_a16w16_skinny(x, weight, out=out, **cfg["config"])
    elif cfg["kind"] == "asm_preshuffle":
        packed = getattr(weight, "_k31_preshuffled", None)
        if packed is None:
            torch.mm(x, weight.t(), out=out)
        else:
            from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm

            gemm_a16w16_asm(
                x,
                packed,
                out,
                splitK=1,
                kernelName="_ZN5aiter43bf16gemm_fp32bf16_tn_160x64_bshuffle_splitkE",
                bpreshuffle=True,
            )
    elif cfg["config"]["split_k"] == 1:
        _flydsl_no_split(x, weight, out, cfg["config"])
    else:
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16

        gemm_a16w16(
            x,
            weight.t(),
            out=out,
            user_kwargs=cfg["config"],
            stream=torch.cuda.current_stream(x.device),
            layout="nt",
        )
    return True


def mla_linear(x, weight):
    if not mla_gemm_covered(x, weight):
        return torch.nn.functional.linear(x, weight)
    out = x.new_empty((x.shape[0], weight.shape[0]))
    if not mla_linear_into(x, weight, out):
        return torch.nn.functional.linear(x, weight)
    return out
