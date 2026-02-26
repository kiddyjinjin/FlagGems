"""
Tests for operations that use vLLM as reference implementation.
NOTE: These tests use direct torch.equal()/torch.allclose() with manual .cpu()
calls instead of gems_assert_equal()/to_cpu(). This is because vLLM functions are
device-only with no CPU reference implementation. The to_cpu() helper asserts
that ref must be on CPU when --ref cpu is passed, which would fail since vLLM
functions always run on device.
"""

import random
from itertools import product
from math import ceil
from typing import Optional

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE


random.seed(42)


def is_vllm_available():
    try:
        import vllm  # noqa: 401

        return True
    except ImportError:
        return False


VLLM_AVAILABLE = is_vllm_available()


# Import vLLM grouped_topk if available
try:
    from vllm._custom_ops import grouped_topk as vllm_grouped_topk

    HAS_VLLM_GROUPED_TOPK = True
except ImportError:
    HAS_VLLM_GROUPED_TOPK = False
    vllm_grouped_topk = None


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


HOPPER_AVAILABLE = is_hopper_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def check_valid_config(n_expert, n_group, topk):
    if n_expert % n_group != 0:
        return False
    return True


def get_grouped_topk_tolerance(dtype, scoring_func, renormalize):
    if dtype == torch.bfloat16:
        return 5e-3, 1e-3
    elif dtype == torch.float16:
        if scoring_func == 1:
            return 1e-3, 1e-4
        else:
            return 5e-3, 1e-3
    else:
        if renormalize:
            return 5e-4, 1e-4
        else:
            return 1e-5, 1e-5


class CutlassScaledMMTestKit:
    num_test_cases = 16 if QUICK_MODE else 32

    @staticmethod
    def _get_all_combinations():
        # these shapes come from the test file of op `cutlass_scaled_mm` of vLLM
        mnk = [
            (1, 256, 128),
            (1, 16384, 1024),
            (1, 24576, 496),
            (16, 256, 496),
            (16, 16384, 128),
            (16, 24576, 4096),
            (32, 8192, 4096),
            (32, 16384, 4096),
            (33, 1024, 1024),
            (33, 8192, 128),
            (64, 2048, 496),
            (64, 16384, 1024),
            (100, 8192, 496),
            (128, 32768, 4096),
            (256, 4096, 4096),
            (512, 256, 1024),
            (512, 8192, 4096),
            (512, 16384, 128),
            (512, 24576, 128),
        ]
        scale_shape_types = ["scalar", "vector", "matrix"]
        if_use_bias = [True, False]
        dtypes = [(torch.int8, torch.float16), (torch.float8_e4m3fn, torch.bfloat16)]

        combinations = product(
            mnk, scale_shape_types, scale_shape_types, if_use_bias, dtypes
        )
        return combinations

    @classmethod
    def _rand_sample(cls, all_params):
        random.shuffle(all_params)
        return all_params[: cls.num_test_cases]

    @classmethod
    def get_test_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            bias,
            (in_dtype, out_dtype),
        ) in combinations:
            is_scalar_or_vector_dequant = a_scale_category in [
                "scalar",
                "vector",
            ] and b_scale_category in ["scalar", "vector"]
            is_block_dequant = (
                a_scale_category == "matrix" and b_scale_category == "matrix"
            )

            if not (is_scalar_or_vector_dequant or is_block_dequant):
                continue

            if is_block_dequant and (bias is not None or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        return cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        elif category == "vector":
            if is_a_scale:
                return (M,)
            else:
                return (N,)
        else:
            if is_a_scale:
                return (M, ceil(K / 128))
            else:
                return (ceil(K / 128), ceil(N / 128))

    @staticmethod
    def baseline_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ):
        def group_broadcast(t: torch.Tensor, shape):
            for i, s in enumerate(shape):
                if t.shape[i] != s and t.shape[i] != 1:
                    assert s % t.shape[i] == 0
                    t = (
                        t.unsqueeze(i + 1)
                        .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                        .flatten(i, i + 1)
                    )
            return t

        scale_a_full = group_broadcast(scale_a, a.shape)
        scale_b_full = group_broadcast(scale_b, b.shape)

        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)

        lhs = scale_a_full * a_f32
        rhs = scale_b_full * b_f32

        output = torch.mm(lhs, rhs).to(out_dtype)

        if bias is not None:
            output = output + bias

        return output


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and HOPPER_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.parametrize("p", CutlassScaledMMTestKit.get_test_params())
def test_cutlass_scaled_mm(p):
    kit = CutlassScaledMMTestKit

    M, N, K = p["M"], p["N"], p["K"]
    in_dtype = p["in_dtype"]
    out_dtype = p["out_dtype"]
    a_scale_category = p["a_scale_category"]
    b_scale_category = p["b_scale_category"]

    if in_dtype == torch.int8:
        a = to_int8(torch.randn((M, K), device=flag_gems.device))
        b = to_int8(
            torch.randn((K, N), device=flag_gems.device).t().contiguous().t() * 5
        )
    else:
        a = to_fp8(torch.randn((M, K), device=flag_gems.device))
        b = to_fp8(torch.randn((K, N), device=flag_gems.device).t().contiguous().t())

    a_scale_shape = kit.get_scale_shape(M, N, K, a_scale_category)
    b_scale_shape = kit.get_scale_shape(M, N, K, b_scale_category, False)

    scale_a = torch.randn(a_scale_shape, device=flag_gems.device, dtype=torch.float32)
    scale_b = torch.randn(b_scale_shape, device=flag_gems.device, dtype=torch.float32)

    scale_a = scale_a.contiguous()
    # convert scale_b to col-major
    # (for scalar/vector scale_b, this's a identical transformation)
    scale_b = scale_b.t().contiguous().t()

    bias = None
    if p["use_bias"]:
        bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

    c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

    flag_gems.cutlass_scaled_mm(c, a, b, scale_a, scale_b, bias)

    output_ref = kit.baseline_scaled_mm(
        a, b, scale_a.view(-1, 1), scale_b.view(1, -1), out_dtype, bias
    )

    if in_dtype == torch.int8:
        rtol, atol = 1e-1, 1.0
    else:
        rtol, atol = 5e-1, 1.5e-1

    torch.testing.assert_close(c, output_ref, rtol=rtol, atol=atol)


# grouped_topk test parameters
N_TOKEN_LIST = [1, 3, 8] if not QUICK_MODE else [8]
N_EXPERT_LIST = [8, 16] if not QUICK_MODE else [16]
N_GROUP_LIST = [2, 4] if not QUICK_MODE else [4]
TOPK_LIST = [1, 2] if not QUICK_MODE else [2]
RENORMALIZE_LIST = [True, False] if not QUICK_MODE else [True]
SCORING_FUNC_LIST = [0, 1] if not QUICK_MODE else [0]
GROUPED_TOPK_DTYPE_LIST = [torch.bfloat16, torch.float32] if not QUICK_MODE else [torch.float32]
LARGE_SCALE_DTYPE_LIST = [torch.float32, torch.bfloat16]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM_GROUPED_TOPK, reason="vLLM grouped_topk is not available")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", N_TOKEN_LIST)
@pytest.mark.parametrize("n_expert", N_EXPERT_LIST)
@pytest.mark.parametrize("n_group", N_GROUP_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
@pytest.mark.parametrize("dtype", GROUPED_TOPK_DTYPE_LIST)
def test_accuracy_grouped_topk(
    n_token,
    n_expert,
    n_group,
    topk,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk accuracy against vLLM CUDA implementation"""
    if not check_valid_config(n_expert, n_group, topk):
        pytest.skip("Invalid config")

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    topk_group = topk
    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = vllm_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    with flag_gems.use_gems():
        res_topk_weights, res_topk_ids = flag_gems.grouped_topk(
            scores.clone(),
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )

    assert torch.equal(res_topk_ids.cpu(), ref_topk_ids.cpu()), "topk_ids mismatch"

    atol, rtol = get_grouped_topk_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(
        res_topk_weights.cpu(), ref_topk_weights.cpu(), atol=atol, rtol=rtol
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM_GROUPED_TOPK, reason="vLLM grouped_topk is not available")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", [32, 64])
@pytest.mark.parametrize("n_expert", [64])
@pytest.mark.parametrize("n_group", [8])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("dtype", LARGE_SCALE_DTYPE_LIST)
def test_accuracy_grouped_topk_large_scale(
    n_token,
    n_expert,
    n_group,
    topk,
    topk_group,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk with larger scale configurations"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = vllm_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    with flag_gems.use_gems():
        res_topk_weights, res_topk_ids = flag_gems.grouped_topk(
            scores.clone(),
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )

    assert torch.equal(res_topk_ids.cpu(), ref_topk_ids.cpu()), "topk_ids mismatch"

    atol, rtol = get_grouped_topk_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(
        res_topk_weights.cpu(), ref_topk_weights.cpu(), atol=atol, rtol=rtol
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM_GROUPED_TOPK, reason="vLLM grouped_topk is not available")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("renormalize", [True, False])
def test_accuracy_grouped_topk_scaling_factor(routed_scaling_factor, renormalize):
    """Test grouped_topk with different scaling factors"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
        )

    assert torch.equal(res_ids.cpu(), ref_ids.cpu()), "topk_ids mismatch"

    atol, rtol = get_grouped_topk_tolerance(dtype, 0, renormalize)
    torch.testing.assert_close(res_weights.cpu(), ref_weights.cpu(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM_GROUPED_TOPK, reason="vLLM grouped_topk is not available")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_accuracy_grouped_topk_single_token(renormalize, scoring_func):
    """Test grouped_topk with single token"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((1, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
        )

    assert torch.equal(res_ids.cpu(), ref_ids.cpu()), "topk_ids mismatch"

    atol, rtol = get_grouped_topk_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(res_weights.cpu(), ref_weights.cpu(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM_GROUPED_TOPK, reason="vLLM grouped_topk is not available")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
def test_accuracy_grouped_topk_sigmoid(renormalize):
    """Test grouped_topk with sigmoid scoring function"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
        )

    assert torch.equal(res_ids.cpu(), ref_ids.cpu()), "topk_ids mismatch"

    atol, rtol = get_grouped_topk_tolerance(dtype, 1, renormalize)
    torch.testing.assert_close(res_weights.cpu(), ref_weights.cpu(), atol=atol, rtol=rtol)
