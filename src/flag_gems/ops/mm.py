import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

try:
    device_id = torch_device_fn.current_device()
except AttributeError:
    device_id = 0

try:
    L2_CACHE_SIZE = torch_device_fn.get_device_properties(device_id).L2_cache_size
    SM_COUNT = torch_device_fn.get_device_properties(device_id).multi_processor_count
except AttributeError:
    L2_CACHE_SIZE = 40 * 1024 * 1024  # 40MB in bytes
    SM_COUNT = 82  # nvidia 3090
CACHE_USAGE_THRESHOLD = 0.7

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit()
def swizzle_tile(
    tile_id,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@libentry()
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
    }
)
@triton.jit
def first_wave(
    A,
    B,
    C,
    M,
    N,
    K,
    locks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    total_full_tiles_streamk,
    total_partial_tiles_streamk,
    iters_per_tile,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)  # pid range from 0 to sm_count
    start_iter = pid * total_full_tiles_streamk + tl.minimum(
        pid, total_partial_tiles_streamk
    )
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(
        pid + 1, total_partial_tiles_streamk
    )
    while start_iter < last_iter:
        remain_iters = start_iter % iters_per_tile
        # Iterate over the K axis. Recalculate end_iter as M/N may change during the iteration.
        end_iter = tl.minimum(start_iter + (iters_per_tile - remain_iters), last_iter)

        tile_id = start_iter // iters_per_tile

        pid_m, pid_n = swizzle_tile(
            tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
        )

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        # pointers
        A_ptr = (
            A
            + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
            + BLOCK_K * stride_ak * remain_iters
        )
        B_ptr = (
            B
            + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
            + BLOCK_K * stride_bk * remain_iters
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_ptr)
                b = tl.load(B_ptr)
            else:
                k_mask = (current_iter % iters_per_tile) * BLOCK_K + rk < K
                _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
                a = tl.load(A_ptr, mask=k_mask[None, :], other=_0)
                b = tl.load(B_ptr, mask=k_mask[:, None], other=_0)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk
        # last iteration of the tile always happens before its start on another SM
        if end_iter % iters_per_tile == 0:
            C_ptr = C + (
                rm[:, None] * stride_cm + rn[None, :] * stride_cn
            )  # compute inside the if/else to avoid spilling!
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_ptr, acc, mask=mask)
            if remain_iters != 0:  # only if tile has been partially processed
                tl.atomic_xchg(locks + tile_id, 1)
        else:
            while tl.atomic_cas(locks + tile_id, 1, 1) != 1:
                pass
            C_ptr = C + (
                rm[:, None] * stride_cm + rn[None, :] * stride_cn
            )  # compute inside the if/else to avoid spilling!
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.atomic_add(C_ptr, acc, mask=mask, sem="relaxed")
        start_iter = end_iter


@libentry()
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
    }
)
@triton.jit
def classic_tiles_mm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    total_tiles_streamk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # first wave has done more tiles than there are SMs, we adjust pid
    tile_id = tl.program_id(0) + total_tiles_streamk
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_iobound") + runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
    }
)
@triton.jit
def mm_kernel_with_grouped_k(
    A,
    B,
    C,  # [Split_K, M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,  # Number of split-K groups
    GROUP_K_LENGTH: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    assert GROUP_K_LENGTH % BLOCK_K == 0, "GROUP_K_LENGTH must be divisible by BLOCK_K"

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    total_num_m = num_blocks_m * SPLIT_K

    pid_n = pid // total_num_m
    odd_column = pid_n % 2
    pid_m_normal = pid % total_num_m
    # this is a line-one implementation for the following code:
    #     if odd_column:
    #         pid_m_for_c = (total_num_m - 1) - pid_m_normal
    #     else:
    #         pid_m_for_c = pid_m_normal
    pid_m_for_c = (1 - odd_column) * pid_m_normal + odd_column * (
        total_num_m - 1 - pid_m_normal
    )

    pid_m = pid_m_for_c % num_blocks_m
    pid_k = pid_m_for_c // num_blocks_m

    # Calculate K_LENGTH based on pid_k
    group_k_length = min(K - pid_k * GROUP_K_LENGTH, GROUP_K_LENGTH)

    # matrix multiplication
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * GROUP_K_LENGTH
    offs_k = k_start + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    # pointers
    A_ptr = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(group_k_length, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
        else:
            k_remaining = k_start + group_k_length - k * BLOCK_K
            a = tl.load(A_ptr, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(B_ptr, mask=offs_k[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    # Store results
    offs_cb = pid_k * stride_cb
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C_ptr = C + offs_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(C_ptr, acc, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("sum"), key=["M", "N"])
@triton.jit
def group_merge_kernel(
    SRC,  # [SPLIT_K, M, N] 3D Tensor
    DST,  # [M, N]
    SPLIT_K,
    M,
    N,
    stride_k,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(SPLIT_K):
        src_ptr = (
            SRC + k * stride_k + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        )
        sub_matrix = tl.load(src_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        acc += sub_matrix
    acc = acc.to(DST.dtype.element_ty)
    dst_ptr = DST + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(dst_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_iobound"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
    }
)
@triton.jit
def mm_kernel_iobound(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # column major tile
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["log", "log", "log", None, None],
)
@triton.jit
def mm_kernel_general(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = prev_multiple + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def streamk_mm(a, b, c, M, N, K, c_dtype, sm_count=108):
    # TODO: profile to different settings for different chip
    if b.stride(0) == 1:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
        num_stages = 3
        num_warps = 8
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        num_stages = 3
        num_warps = 16

    GROUP_M = 8
    number_blocks_m = triton.cdiv(M, BLOCK_M)
    number_blocks_n = triton.cdiv(N, BLOCK_N)

    total_tiles = number_blocks_m * number_blocks_n
    iters_per_tile = triton.cdiv(K, BLOCK_K)
    tiles_per_wave = sm_count

    # tiles that would executed in the last wave in general situation.
    # and this is the tiles that we are going to adopt streamk)
    total_tiles_streamk = total_tiles % tiles_per_wave
    # mini wave
    total_iters_streamk = total_tiles_streamk * iters_per_tile
    total_full_tiles_streamk = total_iters_streamk // tiles_per_wave
    total_partial_tiles_streamk = total_iters_streamk % tiles_per_wave

    locks = torch.zeros((total_tiles_streamk,), device=a.device, dtype=torch.int32)

    with torch_device_fn.device(a.device):
        first_wave[(tiles_per_wave,)](
            a,
            b,
            c,
            M,
            N,
            K,
            locks,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            total_full_tiles_streamk=total_full_tiles_streamk,
            total_partial_tiles_streamk=total_partial_tiles_streamk,
            iters_per_tile=iters_per_tile,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        classic_tiles_mm[(total_tiles - total_tiles_streamk,)](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            total_tiles_streamk=total_tiles_streamk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return c


def splitk_mm(a, b, c, M, N, K, c_dtype):
    GROUP_K_LENGTH = 1024
    SPLIT_K = triton.cdiv(K, GROUP_K_LENGTH)
    # TODO: float32 or c_dtype
    multi_c = torch.empty((SPLIT_K, M, N), device=a.device, dtype=c_dtype)
    # 1st kernel: compute partial results
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * SPLIT_K,
    )
    grid2 = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_with_grouped_k[grid](
            a,
            b,
            multi_c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            multi_c.stride(0),
            multi_c.stride(1),
            multi_c.stride(2),
            SPLIT_K=SPLIT_K,
            GROUP_K_LENGTH=GROUP_K_LENGTH,
        )
        # return torch.sum(multi_c, dim=0)
        # 2nd kernel: merge partial results
        group_merge_kernel[grid2](
            multi_c,
            c,
            SPLIT_K,
            M,
            N,
            multi_c.stride(0),
            multi_c.stride(1),
            multi_c.stride(2)
        )
    return c


def iobound_mm(a, b, c, M, N, K):
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_iobound[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1)
        )
    return c


def general_mm(a, b, c, M, N, K):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_general[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
        )
    return c


def mini_mm_scenario(a, b, l2_cache_size=40 * 1024 * 1024, cache_usage_threshold=0.8):
    return (
        a.shape[0] <= 256
        and (a.numel() * a.element_size() + b.shape[0] * b.element_size())
        < l2_cache_size * cache_usage_threshold
    )


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] > 7).
    # The optimal settings for other devices need to be determined through real testing.
    capability = torch_device_fn.get_device_capability(device_id)
    return (
        capability[0] > 7
        and a.dtype in [torch.float16]
        and b.dtype in [torch.float16]
        and M > 1024
        and N > 1024
        and K > M * 10
    )


def two_stages_splitk_mm_scenario(M, N, K):
    return (M < 32 or N < 32) and (K > M * 10 or K > N * 10)


def mm(a, b):
    logger.debug("GEMS MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)

    if mini_mm_scenario(a, b, L2_CACHE_SIZE, CACHE_USAGE_THRESHOLD):
        return iobound_mm(a, b, c, M, N, K)
    elif streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, c_dtype, sm_count=SM_COUNT)
    elif two_stages_splitk_mm_scenario(M, N, K):
        return splitk_mm(a, b, c, M, N, K, c_dtype)
    else:
        return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = out.dtype
    c = out
    # launch kernel
    if mini_mm_scenario(a, b, L2_CACHE_SIZE, CACHE_USAGE_THRESHOLD):
        return iobound_mm(a, b, c, M, N, K)
    elif streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, c_dtype, sm_count=SM_COUNT)
    elif two_stages_splitk_mm_scenario(M, N, K):
        return splitk_mm(a, b, c, M, N, K, c_dtype)
    else:
        return general_mm(a, b, c, M, N, K)
