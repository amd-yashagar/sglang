# SPDX-License-Identifier: Apache-2.0
"""Opt-in K3.1 grouped verify kernel, using existing AITER inputs/reference.

Run after bench_dsa_paged_indexer.py --build-only, under a gfx950 GPU lease.
SGLANG_DSA_INDEXER_BENCH_DIR must point to the prebuilt benchmark cache.
Set TRITON_CACHE_DIR to its triton-cache subdirectory BEFORE starting pytest.
Set SGLANG_JIT_CACHE_DIR to the CPU-prebuilt top-k cache.
"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from sglang.kernels.ops.attention.dsa.gfx950_mqa_logits import (
    gfx950_flydsl_paged_mqa_logits,
    gfx950_paged_mqa_logits,
    use_flydsl_mqa,
    use_grouped_mqa,
)


@pytest.fixture(scope="module")
def bench():
    if not os.getenv("SGLANG_DSA_INDEXER_HIP_LIB"):
        pytest.skip("Prebuild grouped_mqa.cu and set SGLANG_DSA_INDEXER_HIP_LIB")
    path = (
        Path(__file__).resolve().parents[5]
        / "benchmark/kernels/bench_dsa_paged_indexer.py"
    )
    spec = importlib.util.spec_from_file_location("dsa_indexer_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reject_runtime_compilation()
    module.install_schedule()
    module.helpers = module.load_helpers()
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("gfx950 only")
    return module


def inputs(bench, batch, length, seed=13, next_n=5):
    return bench.helpers.build_inputs(
        (batch, next_n, 32, 128, 64, True),
        length,
        ((length + 255) // 256) * 256,
        seed=seed,
    )


def check(bench, inp, length, *, graph=False, flydsl=False):
    q, cache, weights = inp["q_fp8"], inp["kv_cache"], inp["weights"]
    n = q.shape[1]
    lens, tables = inp["context_lens"], inp["kv_indices"]
    width = inp["max_model_len"]
    tables[:, (length + 63) // 64 :] = -1  # poisoned padding must never be read
    cache_before = cache.clone()
    # Offset output storage checks both ends, including graph replay.
    storage = torch.full((q.shape[0] * n * width + 128,), 123456.0, device="cuda")
    out = storage[64:-64].view(q.shape[0] * n, width)

    def launch():
        kernel = gfx950_flydsl_paged_mqa_logits if flydsl else gfx950_paged_mqa_logits
        return kernel(q, cache, weights, lens, tables, width, out=out)

    launch()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        # Includes empty causal rows (request length 4), short rows, full pages,
        # partial pages, and rows changing their number of pages at replay.
        values = [n - 1, n, 63, 64, 65, 2047, 2048, 2049, 8193, length]
        lens.copy_(
            torch.tensor(
                [min(length, values[i % len(values)]) for i in range(q.shape[0])],
                dtype=torch.int32,
                device="cuda",
            )
        )
        g.replay()
    assert torch.equal(cache, cache_before), "index-K was modified"
    assert (storage[:64] == 123456.0).all() and (storage[-64:] == 123456.0).all()
    row_lens = (
        lens[:, None] - n + 1 + torch.arange(n, device="cuda", dtype=torch.int32)
    ).flatten()
    valid = torch.arange(length, device="cuda")[None, :] < row_lens[:, None]
    reference = bench.helpers.run_torch(inp, length).masked_fill(~valid, 0)
    got = out[:, :length].masked_fill(~valid, 0)
    torch.testing.assert_close(got, reference, rtol=1e-2, atol=5.0)
    error = float((got - reference).abs().max())
    with patch.dict(
        os.environ,
        {"SGLANG_DSA_GFX950_GROUPED_MQA": "0", "SGLANG_DSA_GFX950_FLYDSL_MQA": "0"},
    ):
        baseline = bench.aiter_paged_mqa_logits_verify(
            q.flatten(0, 1),
            cache,
            weights,
            lens,
            tables,
            width,
            q_offset=q.shape[0] * n,
            B=q.shape[0],
            next_n=n,
            kv_block_size=64,
        )[:, :length].masked_fill(~valid, 0)
    baseline_error = float((baseline - reference).abs().max())
    assert error <= baseline_error + 1e-4, "Exceeds existing kernel error"

    if reference.abs().sum():
        assert bench.helpers.calc_diff(got, reference) < 1e-3
    selected = torch.empty((q.shape[0] * n, 2048), dtype=torch.int32, device="cuda")
    plan = bench.plan_topk_v2(row_lens)
    bench.topk_transform_paged_v2(out, row_lens, None, selected, 64, plan)
    bench.topk_check(out, selected, row_lens, "emitted logits", strict=True)
    # gfx950 FP8 MFMA dot rounding changes cutoff membership even in baseline.
    # Strict FP32/FP64 and the hardware FP8 dot are not numerically identical.
    # Exact selector correctness is required above; independent reference swaps
    # must stay inside the mathematically necessary 2*logit-error uncertainty.
    status = bench.topk_check(reference, selected, row_lens, "fp32", strict=False)
    assert status["max_cutoff_delta"] <= 2 * error + 1e-5
    return status


@pytest.mark.parametrize("batch", [1, 8, 16, 64])
@pytest.mark.parametrize("length", [4, 5, 63, 64, 65, 2047, 2048, 2049, 32767, 32769])
def test_grouped_mqa_partial_pages(bench, batch, length):
    check(bench, inputs(bench, batch, length), length)


def test_grouped_mqa_graph_replay_mixed_lengths(bench):
    check(bench, inputs(bench, 16, 32769), 32769, graph=True)


@pytest.mark.parametrize("zero_queries", [False, True])
def test_grouped_mqa_exact_cutoff_ties(bench, zero_queries):
    inp = inputs(bench, 16, 4099)
    if zero_queries:
        inp["q_fp8"].view(torch.uint8).zero_()
    else:
        inp["weights"].zero_()
    assert check(bench, inp, 4099)["exact_mod_ties"]


def test_grouped_mqa_dispatch_and_layout(bench, monkeypatch):
    inp = inputs(bench, 16, 32769)
    args = (
        inp["q_fp8"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["kv_indices"],
        inp["max_model_len"],
    )
    monkeypatch.delenv("SGLANG_DSA_GFX950_GROUPED_MQA", raising=False)
    assert not use_grouped_mqa(*args)
    monkeypatch.setenv("SGLANG_DSA_GFX950_GROUPED_MQA", "1")
    assert use_grouped_mqa(*args)
    bad = list(args)
    bad[2] = torch.empty((80, 64), device="cuda")[:, ::2]
    assert not use_grouped_mqa(*bad)
    with pytest.raises(ValueError):
        gfx950_paged_mqa_logits(*bad)
    for batch in (1, 8):
        small = inputs(bench, batch, 32769)
        assert not use_grouped_mqa(
            small["q_fp8"],
            small["kv_cache"],
            small["weights"],
            small["context_lens"],
            small["kv_indices"],
            small["max_model_len"],
        )


def test_grouped_mqa_empty_and_bad_output(bench, monkeypatch):
    inp = inputs(bench, 16, 65)
    args = (
        inp["q_fp8"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["kv_indices"],
        inp["max_model_len"],
    )
    with pytest.raises(ValueError):
        gfx950_paged_mqa_logits(
            *args, out=torch.empty((80, 256), device="cuda", dtype=torch.float16)
        )
    empty = list(args)
    for i in (0, 2, 3, 4):
        empty[i] = empty[i][:0]
    monkeypatch.delenv("SGLANG_DSA_INDEXER_HIP_LIB", raising=False)
    assert gfx950_paged_mqa_logits(*empty).shape == (0, 256)
    with pytest.raises(RuntimeError, match="Prebuild"):
        gfx950_paged_mqa_logits(*args)


def test_membership_checker_rejects_missing_strict_winner(bench):
    scores = torch.ones((1, 2050), device="cpu")
    scores[:, :2047] = 2
    selected = torch.cat(
        (torch.arange(2046, device="cpu"), torch.tensor([2047, 2048], device="cpu"))
    )[None, :]
    status = bench.topk_check(
        scores, selected, torch.tensor([2050], device="cpu"), "checker", strict=False
    )
    assert status["below_cutoff_count"] == 0
    assert status["missing_strict_winners"] == 1
    assert not status["exact_mod_ties"]
    with pytest.raises(AssertionError, match="missing strict winner"):
        bench.topk_check(
            scores, selected, torch.tensor([2050], device="cpu"), "checker", strict=True
        )


@pytest.mark.parametrize("batch", [1, 8, 16, 64])
@pytest.mark.parametrize("next_n", [1, 5])
@pytest.mark.parametrize("length", [65, 32767, 32769, 131073])
def test_flydsl_mqa_partial_pages(bench, batch, next_n, length):
    check(bench, inputs(bench, batch, length, next_n=next_n), length, flydsl=True)


@pytest.mark.parametrize("next_n", [1, 5])
def test_flydsl_mqa_graph_replay_mixed_lengths(bench, next_n):
    check(
        bench, inputs(bench, 16, 32769, next_n=next_n), 32769, graph=True, flydsl=True
    )


@pytest.mark.parametrize("zero_queries", [False, True])
def test_flydsl_mqa_exact_cutoff_ties(bench, zero_queries):
    inp = inputs(bench, 16, 4099)
    inp["q_fp8"].view(torch.uint8).zero_() if zero_queries else inp["weights"].zero_()
    assert check(bench, inp, 4099, flydsl=True)["exact_mod_ties"]


def test_flydsl_mqa_dispatch_and_layout(bench, monkeypatch):
    inp = inputs(bench, 16, 32769)
    args = [
        inp[k]
        for k in (
            "q_fp8",
            "kv_cache",
            "weights",
            "context_lens",
            "kv_indices",
            "max_model_len",
        )
    ]
    monkeypatch.delenv("SGLANG_DSA_GFX950_FLYDSL_MQA", raising=False)
    assert not use_flydsl_mqa(*args)
    monkeypatch.setenv("SGLANG_DSA_GFX950_FLYDSL_MQA", "1")
    assert use_flydsl_mqa(*args)
    bad = list(args)
    bad[2] = torch.empty((80, 64), device="cuda")[:, ::2]
    assert not use_flydsl_mqa(*bad)
    with pytest.raises(ValueError):
        gfx950_flydsl_paged_mqa_logits(*bad)
    for batch, n, length, expected in (
        (1, 1, 32768, False),
        (16, 5, 1024, False),
        (8, 1, 32768, True),
        (1, 5, 131072, True),
        (64, 5, 131072, False),
    ):
        x = inputs(bench, batch, length, next_n=n)
        assert (
            use_flydsl_mqa(
                *(
                    x[k]
                    for k in (
                        "q_fp8",
                        "kv_cache",
                        "weights",
                        "context_lens",
                        "kv_indices",
                        "max_model_len",
                    )
                )
            )
            is expected
        )


@pytest.mark.parametrize("flydsl", [False, True])
def test_gfx950_mfma_rounding_is_not_fp32_dot(bench, flydsl):
    # 256 plus 127 copies of 2**-6: the seven small products sharing the
    # first MFMA group are lost. This is hardware dot rounding, not top-k ties
    # or a dequantization/reduction-order effect. All operands/scales are exact.
    inp = inputs(bench, 1, 65)
    q, cache, w = inp["q_fp8"], inp["kv_cache"], inp["weights"]
    q.view(torch.uint8).fill_(0x38)  # E4M3FN 1.0
    flat = cache.view(cache.shape[0], -1)
    flat[:, :8192].fill_(0x08)  # E4M3FN 2**-6
    flat[:, :8192].view(-1, 4, 8, 16, 16)[:, :, 0, :, 0] = 0x78  # E4M3FN 256
    flat[:, 8192:].view(torch.float32).fill_(1)
    w.zero_()
    w[:, 0] = 1
    kernel = gfx950_flydsl_paged_mqa_logits if flydsl else gfx950_paged_mqa_logits
    got = kernel(
        q, cache, w, inp["context_lens"], inp["kv_indices"], inp["max_model_len"]
    )
    assert bool((got[:, :61] == 257.875).all())
    assert 256 + 127 * 2**-6 == 257.984375


def test_flydsl_mqa_empty_and_bad_output(bench, monkeypatch):
    inp = inputs(bench, 16, 65)
    args = [
        inp[k]
        for k in (
            "q_fp8",
            "kv_cache",
            "weights",
            "context_lens",
            "kv_indices",
            "max_model_len",
        )
    ]
    with pytest.raises(ValueError):
        gfx950_flydsl_paged_mqa_logits(
            *args, out=torch.empty((80, 256), dtype=torch.float16, device="cuda")
        )
    for i in (0, 2, 3, 4):
        args[i] = args[i][:0]
    monkeypatch.delenv("SGLANG_DSA_INDEXER_HIP_LIB", raising=False)
    assert gfx950_flydsl_paged_mqa_logits(*args).shape == (0, 256)
