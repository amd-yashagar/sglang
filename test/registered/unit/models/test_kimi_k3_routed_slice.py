"""The K3 latent view may skip the pack only inside Aiter's stage-1 quant cutoff."""

import unittest

import torch

from sglang.srt.models.kimi_k3 import (
    _AITER_STAGE1_FUSED_QUANT_ROWS,
    _routed_slice_aiter_quant_ready,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _aligned_slice(tokens: int, latent: int, leading: int) -> torch.Tensor:
    fused = torch.empty(tokens, leading + latent, dtype=torch.bfloat16)
    return fused[:, leading:]


class TestRoutedSliceAiterQuantReady(CustomTestCase):
    def test_matches_aiter_stage1_cutoff(self):
        # Aiter uses M <= 8*256/topk, with true division. The predicate has to
        # agree on both sides of that boundary, including topk that does not
        # divide 2048.
        for topk in (1, 7, 16):
            limit = _AITER_STAGE1_FUSED_QUANT_ROWS / topk
            below = int(limit)
            above = below + 1
            ready = _aligned_slice(above, 32, 32)
            self.assertTrue(_routed_slice_aiter_quant_ready(ready[:below], below, topk))
            self.assertFalse(_routed_slice_aiter_quant_ready(ready, above, topk))

    def test_rejects_misaligned_pointer_or_stride(self):
        storage = torch.empty(64, dtype=torch.bfloat16)
        misaligned = storage[1:33].view(1, 32)
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        self.assertFalse(_routed_slice_aiter_quant_ready(misaligned, 1, 1))

        odd_stride = _aligned_slice(1, 17, 0)
        self.assertFalse(_routed_slice_aiter_quant_ready(odd_stride, 1, 1))

    def test_rejects_non_positive_topk(self):
        row = _aligned_slice(1, 32, 32)
        self.assertFalse(_routed_slice_aiter_quant_ready(row, 1, 0))


if __name__ == "__main__":
    unittest.main()
