"""The gfx950 DFLASH MLA draft verify against extend_attention_fwd."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla_draft import (
    DraftVerifyMLA,
    verify_mla_draft_fwd,
)
from sglang.srt.layers.attention.triton_draft_hip import init_draft_backend
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the DFLASH MLA draft verify serves the gfx950 monokernel only.
register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "the DFLASH MLA draft verify is gfx950's"
)
class TestDraftVerifyMLA(CustomTestCase):
    def test_dflash_mla_kernel_does_not_require_monokernel_context(self):
        backend = SimpleNamespace(
            use_mla=True,
            topk=1,
            dcp_size=1,
            draft_bf16_dot=False,
            runs_decode=True,
        )
        dflash_utils = ModuleType("sglang.srt.speculative.dflash_utils")

        def parse_dflash_draft_config(**kwargs):
            return SimpleNamespace(is_mla=True)

        dflash_utils.parse_dflash_draft_config = parse_dflash_draft_config

        def disable(fn):
            return fn

        runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
        with (
            patch.dict(
                sys.modules,
                {"sglang.srt.speculative.dflash_utils": dflash_utils},
            ),
            patch.object(torch.compiler, "disable", disable),
        ):
            init_draft_backend(backend, runner)
        self.assertTrue(backend.draft_bf16_dot)
        self.assertFalse(backend.runs_decode)
        self.assertIsNotNone(backend.verify_mla_draft_fwd)

    def test_large_batch_uses_more_prefix_splits(self):
        verifier = object.__new__(DraftVerifyMLA)
        verifier.n_head_blocks = 6
        verifier.max_splits = 128
        self.assertEqual(verifier._num_splits(16), 5)
        self.assertEqual(verifier._num_splits(64), 4)

    def test_tiny_bf16_dots_match_fp32_reference(self):
        """A tiny one-row case keeps probability and split-partial rounding visible."""
        torch.manual_seed(0)
        prefix, d, dv = 7, 576, 512
        scale, k_scale, v_scale = 192**-0.5, 0.7, 1.3
        q = torch.randn(1, 1, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 1, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 1, dv, dtype=torch.bfloat16, device="cuda")
        k_buffer = torch.randn(prefix + 3, 1, d, device="cuda").to(torch.float8_e4m3fn)
        v_buffer = torch.randn(prefix + 3, 1, dv, device="cuda").to(torch.float8_e4m3fn)
        kv_indices = torch.randperm(prefix + 3, device="cuda")[:prefix]
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device="cuda")
        actual = torch.empty(1, 1, dv, dtype=torch.bfloat16, device="cuda")

        ran = verify_mla_draft_fwd(
            q,
            k,
            v,
            actual,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            1,
            k_scale,
            v_scale,
            sm_scale=scale,
            sliding_window_size=1024,
            max_bs=1,
            bf16_dot=True,
        )
        self.assertTrue(ran)

        keys = torch.cat(
            [
                k_buffer[kv_indices.long(), 0].float() * k_scale,
                k[0, 0].float().unsqueeze(0),
            ]
        )
        values = torch.cat(
            [
                v_buffer[kv_indices.long(), 0].float() * v_scale,
                v[0, 0].float().unsqueeze(0),
            ]
        )
        probs = (keys @ q[0, 0].float() * scale).softmax(0)
        expected = (probs @ values).to(torch.bfloat16)
        torch.testing.assert_close(actual[0, 0], expected, atol=5e-3, rtol=5e-3)

        error = actual.float() - expected.float()
        scale_cosine = (error * expected.float()).sum().abs() / (
            error.norm() * expected.float().norm()
        ).clamp_min(1e-12)
        self.assertLess(scale_cosine.item(), 0.1)

    def test_matches_extend_attention(self):
        """A causal 6-token block (three live 2-row combine tiles) over a 300-key FP8
        prefix that the 256-key window trims, with BF16 dots."""
        torch.manual_seed(0)
        prefix, l_ext, h_q, d, dv, window = 300, 6, 4, 576, 512, 256
        q = torch.randn(l_ext, h_q, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(l_ext, 1, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(l_ext, 1, dv, dtype=torch.bfloat16, device="cuda")
        # the prefix sits scattered in a cache with spare rows
        k_buffer = torch.randn(prefix + 64, 1, d, device="cuda").to(torch.float8_e4m3fn)
        v_buffer = torch.randn(prefix + 64, 1, dv, device="cuda").to(
            torch.float8_e4m3fn
        )
        kv_indices = torch.randperm(prefix + 64, device="cuda")[:prefix]
        qo_indptr = torch.tensor([0, l_ext], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device="cuda")
        cache = (k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices)
        mask = (None, True, None, l_ext, 1.0, 1.0)
        opts = dict(sm_scale=d**-0.5, sliding_window_size=window, bf16_dot=True)

        actual = torch.empty(l_ext, h_q, dv, dtype=torch.bfloat16, device="cuda")
        ran = verify_mla_draft_fwd(q, k, v, actual, *cache, *mask, max_bs=1, **opts)
        self.assertTrue(ran)
        expected = torch.empty_like(actual)
        extend_attention_fwd(q, k, v, expected, *cache, *mask, **opts)
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
