import unittest

import torch
from sglang.kernels.ops.attention.attn_res import attn_res_fused_tma
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


class TestAttentionResidual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_attn_residual_and_prefix_write(self):
        generator = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device="cuda")

        num_tokens, num_bank_rows, num_valid_bank_rows = 5, 8, 5
        prefix = randn(num_tokens, _HIDDEN_SIZE).to(torch.bfloat16)
        bank = randn(num_tokens, num_bank_rows, _HIDDEN_SIZE).to(torch.bfloat16)
        combine_weight = (randn(_HIDDEN_SIZE) * _HIDDEN_SIZE**-0.5).to(torch.bfloat16)
        output_weight = (1 + 0.1 * randn(_HIDDEN_SIZE)).to(torch.bfloat16)
        output = torch.empty_like(prefix)

        rows = torch.cat(
            [
                bank[:, :num_valid_bank_rows].float(),
                prefix.unsqueeze(1).float(),
            ],
            dim=1,
        )
        rms = torch.rsqrt(rows.square().mean(-1) + 1e-6)
        scores = (rows * combine_weight.float()).sum(-1) * rms
        mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows).sum(1)
        expected = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * output_weight.float()
        )

        attn_res_fused_tma(
            prefix,
            bank,
            combine_weight,
            output_weight,
            output,
            num_valid_bank_rows,
            1e-6,
            write_prefix=True,
        )

        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=4e-2)
        self.assertTrue(torch.equal(bank[:, num_valid_bank_rows], prefix))


class TestAttentionResidualRocm(unittest.TestCase):
    """Same score/softmax/mix/norm reference as the TMA test above, with
    bf16 prefix rounding and fp64 statistics for the native HIP variant.
    Run explicitly on gfx950; the default CI class above remains SM100-only.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm is required")
        from sglang.kernels.ops.attention.attn_res_smallm_hip import attn_res_smallm_hip

        cls.kernel = staticmethod(attn_res_smallm_hip)
        cls.metrics = []
        cls.main_kernel = None
        import os

        baseline = os.environ.get("ATTN_RES_MAIN_BASELINE")
        if baseline:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "attnres_main_baseline", baseline
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.main_kernel = staticmethod(module.attn_res_hip)

    @classmethod
    def tearDownClass(cls):
        import json
        import os
        from pathlib import Path

        path = os.environ.get("ATTN_RES_METRICS_FILE")
        if path:
            Path(path).write_text(json.dumps(cls.metrics, indent=2))

    def check_case(
        self,
        m,
        n,
        h=7168,
        *,
        add=False,
        cancel=False,
        options=False,
        norm=True,
        zero=False,
        saturate=False,
    ):
        g = torch.Generator(device="cuda").manual_seed(1700 + m + n + h)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device="cuda")

        # Strided rows plus a non-power-of-two H exercise masked vector tails.
        prefix = rand(m, h + 8).bfloat16()[:, :h]
        bank = rand(m, 9, h + 8).bfloat16()[:, :, :h]
        cw = rand(h) * h**-0.5
        ow = (1 + 0.1 * rand(h)).bfloat16() if norm else None
        addend = rand(m, h + 8).bfloat16()[:, :h] if add else None
        if zero:
            prefix.zero_()
            bank.zero_()
        if saturate:
            ow.mul_(1000)
        if cancel:
            # Equal, dyadic weights; opposite large rows leave only a tiny
            # perturbed element. A float32 Gram norm loses this term.
            cw.zero_()
            for k in range(0, n - 1, 2):
                bank[:, k + 1].copy_(-bank[:, k])
            prefix.copy_(-bank[:, n - 1])
            prefix[:, 0] = torch.nextafter(
                prefix[:, 0], torch.full_like(prefix[:, 0], float("inf"))
            )
        rounded = prefix if not add else (prefix.float() + addend.float()).bfloat16()
        rows = torch.cat((bank[:, :n].double(), rounded[:, None].double()), 1)
        eps = 1e-5

        def reference(dtype):
            x = rows.to(dtype)
            scores = (x * cw.to(dtype)).sum(-1) * torch.rsqrt(x.square().mean(-1) + eps)
            probs = scores.softmax(-1)
            mix = (probs[..., None] * x).sum(1)
            out = (
                mix
                if ow is None
                else mix
                * torch.rsqrt(mix.square().mean(-1, keepdim=True) + eps)
                * ow.to(dtype)
            )
            return out, mix, probs

        ref, mix, probs = reference(torch.float64)
        ref32, _, _ = reference(torch.float32)
        gram_error32 = None
        if cancel:
            gram = rows @ rows.transpose(1, 2)
            asq = (probs[:, :, None] * gram * probs[:, None, :]).sum((1, 2))
            direct = mix.square().sum(-1)
            torch.testing.assert_close(asq, direct, atol=1e-9, rtol=1e-7)
            gram32 = rows.float() @ rows.float().transpose(1, 2)
            asq32 = (
                probs.float()[:, :, None] * gram32 * probs.float()[:, None, :]
            ).sum((1, 2))
            gram_error32 = (asq32.double() - direct).abs().max().item()
        outs = []
        streams = []
        for native in (False, True):
            out = torch.empty_like(prefix)
            po = torch.empty_like(prefix)
            stream = torch.empty_like(prefix)
            out8 = torch.empty((m, h), device="cuda", dtype=torch.float8_e4m3fn)
            kw = dict(addend=addend, prefix_out=po, native_math=native)
            if options:
                kw.update(write_prefix=True, stream_out=stream, out_fp8=out8)
            self.kernel(prefix, bank, cw, ow, out, n, eps, eps, **kw)
            if add:
                self.assertTrue(torch.equal(po, rounded))
            if options:
                self.assertTrue(torch.equal(bank[:, n], rounded))
                expected = out.float().clamp(-448, 448).to(out8.dtype)
                self.assertTrue(
                    torch.equal(out8.view(torch.uint8), expected.view(torch.uint8))
                )
                streams.append((stream.double() - mix).abs().max().item())
            torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=4e-2)
            outs.append(out)
        errors = [(o.double() - ref).abs().max().item() for o in outs]
        main_error = None
        if self.main_kernel is not None and h == 7168:
            main_out = torch.empty((m, h), device="cuda", dtype=prefix.dtype)
            main_po = torch.empty_like(main_out)
            main_prefix = torch.empty_like(main_out).copy_(prefix)
            main_add = torch.empty_like(main_out).copy_(addend) if add else None
            self.main_kernel(
                main_prefix,
                bank.contiguous(),
                cw,
                ow,
                main_out,
                n,
                eps,
                eps,
                addend=main_add,
                prefix_out=main_po,
            )
            main_error = (main_out.double() - ref).abs().max().item()
            self.assertLessEqual(errors[1], main_error + 1e-12)
        # No tolerance relaxation: maximum absolute error must not exceed
        # the PR kernel against the very same unrounded fp64 reference.
        self.assertLessEqual(errors[1], errors[0] + 1e-12)
        if options:
            self.assertLessEqual(streams[1], streams[0] + 1e-12)
        self.metrics.append(
            dict(
                M=m,
                nvb=n,
                H=h,
                add=add,
                cancel=cancel,
                options=options,
                norm=norm,
                zero=zero,
                saturate=saturate,
                gram_fp32_error=gram_error32,
                main_max_abs=main_error,
                pr_max_abs=errors[0],
                native_max_abs=errors[1],
                native_fp32_max_abs=(outs[1].float() - ref32).abs().max().item(),
                pr_rmse=(outs[0].double() - ref).square().mean().sqrt().item(),
                native_rmse=(outs[1].double() - ref).square().mean().sqrt().item(),
            )
        )

    def test_native_all_tokens_and_bank_rows(self):
        for m in (1, 5, 8, 40, 64, 320):
            for n in range(1, 9):
                with self.subTest(M=m, nvb=n):
                    self.check_case(m, n)

    def test_native_fused_outputs(self):
        for m in (1, 5, 8, 40, 64, 320):
            for n in (1, 4, 8):
                with self.subTest(M=m, nvb=n):
                    self.check_case(m, n, add=True, options=True)

    def test_native_partial_tiles(self):
        for m in (1, 5, 8, 40, 64, 320):
            for h in (1032, 7160, 7176):
                with self.subTest(M=m, H=h):
                    self.check_case(m, 3, h, add=True, options=True)

    def test_native_cancellation_fp64_gram(self):
        for m in (1, 5, 8, 40, 64, 320):
            for n in (1, 7):
                with self.subTest(M=m, nvb=n):
                    self.check_case(m, n, cancel=True, options=True)

    def test_native_large_m_pending_add(self):
        for n in range(1, 9):
            with self.subTest(nvb=n):
                self.check_case(320, n, add=True)

    def test_native_zero_and_saturation(self):
        self.check_case(64, 8, zero=True, options=True)
        self.check_case(5, 3, saturate=True, options=True)

    def test_native_unnormalized_stream(self):
        for n in range(1, 9):
            with self.subTest(nvb=n):
                self.check_case(8, n, norm=False, options=True)


if __name__ == "__main__":
    unittest.main()
