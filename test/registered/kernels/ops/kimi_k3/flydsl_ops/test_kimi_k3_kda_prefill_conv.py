"""Correctness: FlyDSL KDA prefill conv vs Triton on gfx950."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytest.importorskip("flydsl")
from aiter.jit.utils.chip_info import get_gfx

from sglang.kernels.ops.attention import kda_prefill_conv_aiter_hip
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _gfx950(), reason="gfx950 FlyDSL required")

_DEVICE = torch.device("cuda")


@pytest.mark.parametrize("tokens", [17, 64, 128])
def test_flydsl_prefill_conv_matches_triton(tokens: int, monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_K3_KDA_FUSED_BACKEND", "aiter")
    q_dim = k_dim = v_dim = 128
    dim = q_dim + k_dim + v_dim
    width = 4
    generator = torch.Generator(device=_DEVICE).manual_seed(20260922 + tokens)
    mixed = torch.randn(
        (tokens, dim), dtype=torch.bfloat16, device=_DEVICE, generator=generator
    )
    weight = 0.1 * torch.randn(
        (dim, width), dtype=torch.float32, device=_DEVICE, generator=generator
    )
    conv_ref = torch.randn(
        (2, dim, width - 1),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    conv_act = conv_ref.clone()
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=_DEVICE)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=_DEVICE)

    x = mixed.transpose(0, 1)
    assert kda_prefill_conv_aiter_hip.covered(
        x,
        weight,
        conv_act,
        query_start_loc,
        cache_indices,
        has_initial_state,
        q_dim,
        k_dim,
        v_dim,
    )
    q, k, v = kda_prefill_conv_aiter_hip.run(
        x=x,
        weight=weight,
        bias=None,
        conv_states=conv_act,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    packed = causal_conv1d_fn(
        mixed.transpose(0, 1),
        weight,
        None,
        activation="silu",
        conv_states=conv_ref,
        has_initial_state=has_initial_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        seq_lens_cpu=[tokens],
    ).transpose(0, 1)
    q_ref, k_ref, v_ref = packed.split([q_dim, k_dim, v_dim], dim=-1)
    torch.cuda.synchronize()
    for actual, reference in ((q, q_ref), (k, k_ref), (v, v_ref)):
        denom = reference.float().norm().clamp_min(1e-6)
        rel = (actual.float() - reference.float()).norm() / denom
        assert float(rel) < 1e-3, rel
    assert torch.equal(conv_act, conv_ref)


def test_prefill_conv_backend_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_K3_KDA_FUSED_BACKEND", raising=False)
    assert not kda_prefill_conv_aiter_hip.enabled()
    monkeypatch.setenv("SGLANG_K3_KDA_FUSED_BACKEND", "aiter")
    assert kda_prefill_conv_aiter_hip.enabled()
