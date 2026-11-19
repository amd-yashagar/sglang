"""K3.1 MLA BF16 tests, extending the V-absorb and double-rounded gate contract."""

import unittest

import torch
from sglang.kernels.ops.attention.dsa.mla_v_absorb import (
    mla_q_absorb_cat,
    mla_v_absorb,
    mla_v_absorb_gate,
)
from sglang.srt.models.kimi_k3_amd.mla_fuse import norm_cache
from sglang.srt.models.kimi_k3_amd.mla_gemm import (
    mla_gemm_covered,
    mla_linear,
    mla_linear_into,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")


def rms(x, w, eps):
    z = x.float()
    return (
        z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps) * w.float()
    ).bfloat16()


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm required")
class TestK31MLAFuse(unittest.TestCase):
    def test_norm_cache(self):
        torch.manual_seed(7)
        for m in (0, 1, 5, 40, 320):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(m=m, dtype=dtype):
                    # Real fused_a row stride, with deliberately nonuniform gamma.
                    a = torch.randn(m, 2112, device="cuda", dtype=torch.bfloat16)
                    qw = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
                    kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
                    loc = torch.arange(m, device="cuda", dtype=torch.int64) * 2 + 1
                    if m > 1:
                        loc[-1] = 0
                    cache = torch.full(
                        (2 * m + 3, 1, 576), 7, device="cuda", dtype=dtype
                    )
                    q, kv = norm_cache(
                        a[:, :1536], a[:, 1536:], qw, kw, cache, loc, 1e-5, 2e-5
                    )
                    eq = rms(a[:, :1536], qw, 1e-5)
                    ek = torch.cat((rms(a[:, 1536:2048], kw, 2e-5), a[:, 2048:]), -1)
                    torch.testing.assert_close(q, eq, rtol=8e-3, atol=1e-4)
                    torch.testing.assert_close(kv, ek, rtol=8e-3, atol=1e-4)
                    live = loc > 0
                    # Cache must exactly match this kernels BF16 output rounded on write.
                    torch.testing.assert_close(
                        cache[loc[live], 0].float(),
                        kv[live].to(dtype).float(),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(torch.all(cache[0].float() == 7))
                    self.assertTrue(torch.all(cache[2::2].float() == 7))

    def test_merged_strides(self):
        torch.manual_seed(10)
        for m in (5, 40, 320):
            a = torch.randn(m, 3648, device="cuda", dtype=torch.bfloat16)
            qw = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            loc = torch.arange(m, device="cuda", dtype=torch.int64) + 1
            loc[-1] = -1
            loc[-2] = 4096
            cache = torch.full((4096, 1, 576), 7, device="cuda", dtype=torch.bfloat16)
            q, k = norm_cache(a[:, :1536], a[:, 1536:2112], qw, kw, cache, loc)
            torch.testing.assert_close(
                q, rms(a[:, :1536], qw, 1e-5), rtol=8e-3, atol=1e-4
            )
            ref = torch.cat((rms(a[:, 1536:2048], kw, 1e-5), a[:, 2048:2112]), -1)
            torch.testing.assert_close(k, ref, rtol=8e-3, atol=1e-4)
            torch.testing.assert_close(cache[loc[:-2], 0], k[:-2], rtol=0, atol=0)
            self.assertTrue(torch.all(cache[0] == 7))
            x = torch.randn(m, 12, 512, device="cuda", dtype=torch.bfloat16)
            w = (
                torch.randn(12, 128, 512, device="cuda", dtype=torch.bfloat16)
                / 512**0.5
            ).transpose(1, 2)
            v = torch.bmm(x.transpose(0, 1), w).transpose(0, 1).contiguous().flatten(1)
            torch.testing.assert_close(
                mla_v_absorb_gate(x, w, a[:, 2112:]),
                v * torch.sigmoid(a[:, 2112:]),
                rtol=1.6e-2,
                atol=1.6e-2,
            )

    def test_gemm_routes_and_fallbacks(self):
        torch.manual_seed(11)
        for m in (5, 40, 320):
            for n, k in (
                (2112, 7168),
                (2304, 1536),
                (1536, 7168),
                (7168, 1536),
                (3648, 7168),
            ):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
                ref = torch.nn.functional.linear(x, w)
                out = torch.empty_like(ref)
                self.assertTrue(mla_linear_into(x, w, out))
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=3e-2)
                self.assertFalse(mla_gemm_covered(x.unsqueeze(0), w))
                self.assertFalse(mla_linear_into(x, w, out.float()))
                torch.testing.assert_close(
                    mla_linear(x.unsqueeze(0), w), ref.unsqueeze(0)
                )
                if n == 3648:
                    torch.testing.assert_close(
                        out[:, :2112],
                        torch.nn.functional.linear(x, w[:2112]),
                        rtol=2e-2,
                        atol=3e-2,
                    )
                    torch.testing.assert_close(
                        out[:, 2112:],
                        torch.nn.functional.linear(x, w[2112:]),
                        rtol=2e-2,
                        atol=3e-2,
                    )

    def test_model_merge_reload_and_lazy_gate(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

        def layer(n, k):
            mod = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
            mod.weight.requires_grad_(False)
            mod.quant_method = UnquantizedLinearMethod()
            return mod

        owner = SimpleNamespace(
            _k31_mla_fuse=True,
            use_output_gate=True,
            dcp_size=1,
            has_fused_proj=True,
            _mla_a_gate_weight=None,
            _mla_gate_precomputed=None,
            fused_qkv_a_proj_with_mqa=layer(2112, 7168),
            g_proj=layer(1536, 7168),
            q_b_proj=layer(2304, 1536),
            o_proj=layer(7168, 1536),
        )
        with patch(
            "sglang.srt.runtime_context.get_lora",
            return_value=SimpleNamespace(enable_lora=False, lora_paths=None),
        ):
            KimiK3MLAAttention._merge_mla_a_gate_weights(owner)
            ptr = owner._mla_a_gate_weight.data_ptr()
            self.assertEqual(ptr, owner.fused_qkv_a_proj_with_mqa.weight.data_ptr())
            self.assertEqual(
                owner._mla_a_gate_weight[2112:].data_ptr(),
                owner.g_proj.weight.data_ptr(),
            )
            x = torch.randn(5, 7168, device="cuda", dtype=torch.bfloat16)
            with patch(
                "sglang.srt.layers.layer_boundary.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ):
                latent = KimiK3MLAAttention.prepare_qkv_latent(
                    owner,
                    x,
                    SimpleNamespace(
                        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True)
                    ),
                )
            KimiK3MLAAttention._fork_output_gate(owner, x)
            self.assertIsNone(owner._gate_pending_stream)
            gate = KimiK3MLAAttention._compute_output_gate(owner, x)
            self.assertIsNone(owner._mla_gate_precomputed)
            self.assertEqual(latent.stride(0), 3648)
            self.assertEqual(gate.stride(0), 3648)
            torch.testing.assert_close(
                latent,
                torch.nn.functional.linear(x, owner.fused_qkv_a_proj_with_mqa.weight),
                rtol=2e-2,
                atol=3e-2,
            )
            torch.testing.assert_close(
                gate,
                torch.nn.functional.linear(x, owner.g_proj.weight),
                rtol=2e-2,
                atol=3e-2,
            )
            owner.g_proj.weight.data.fill_(0.25)
            KimiK3MLAAttention._merge_mla_a_gate_weights(owner)
            self.assertEqual(owner._mla_a_gate_weight.data_ptr(), ptr)
            self.assertTrue(torch.all(owner._mla_a_gate_weight[2112:] == 0.25))

    def test_q_absorb_cat(self):
        torch.manual_seed(9)
        for m in (0, 1, 5, 40, 320):
            q = torch.randn(m, 12, 192, device="cuda", dtype=torch.bfloat16)
            w = (
                torch.randn(12, 512, 128, device="cuda", dtype=torch.bfloat16)
                / 128**0.5
            ).transpose(1, 2)
            qn = torch.bmm(q[:, :, :128].transpose(0, 1), w).transpose(0, 1)
            ref = torch.cat((qn, q[:, :, 128:]), -1)
            got = mla_q_absorb_cat(q, w)
            torch.testing.assert_close(got, ref, rtol=8e-3, atol=8e-3)
            self.assertTrue(torch.equal(got[:, :, 512:], q[:, :, 128:]))
            self.assertTrue(got.is_contiguous())

    def test_v_absorb_gate(self):
        torch.manual_seed(8)
        # Runtime stores W as [H,N,K].transpose(1,2), not contiguous [H,K,N].
        for m in (0, 1, 5, 40, 320):
            x = torch.randn(m, 12, 512, device="cuda", dtype=torch.bfloat16)
            w = (
                torch.randn(12, 128, 512, device="cuda", dtype=torch.bfloat16)
                / 512**0.5
            ).transpose(1, 2)
            gate = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16) * 6
            v = torch.bmm(x.transpose(0, 1), w).transpose(0, 1).contiguous()
            ref = v.flatten(1) * torch.sigmoid(gate)
            got = mla_v_absorb_gate(x, w, gate)
            torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=1.6e-2)
            self.assertTrue(got.is_contiguous())
            if 0 < m <= 64:
                # Regression: the existing unfused API retains its semantics.
                wc = w.contiguous()
                out = torch.empty_like(v)
                mla_v_absorb(x, wc, out)
                torch.testing.assert_close(out, v, rtol=8e-3, atol=8e-3)


if __name__ == "__main__":
    unittest.main()
