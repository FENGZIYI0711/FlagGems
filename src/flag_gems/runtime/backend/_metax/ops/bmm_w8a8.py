# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import threading
import weakref

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm_w8a8 import bmm_w8a8 as triton_bmm_w8a8
from flag_gems.ops.bmm_w8a8 import validate_inputs
from flag_gems.utils import libentry


class NativeState(threading.local):
    def __init__(self):
        self.operators = {}
        self.blas = {}


_native_state = NativeState()


def native_gemm():
    device = torch.cuda.current_device()
    if device not in _native_state.operators:
        try:
            from mctlassEx import ScaledGEMM
        except ModuleNotFoundError as error:
            if error.name != "mctlassEx":
                raise
            _native_state.operators[device] = None
        else:
            _native_state.operators[device] = ScaledGEMM()
    return _native_state.operators[device]


class NativeBlas:
    """One handle per host thread and device; no tensor data is cached."""

    def __init__(self):
        self.library = ctypes.CDLL("libmcblas.so")
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.library.mcblasCreate.argtypes = [ctypes.POINTER(pointer)]
        self.library.mcblasDestroy.argtypes = [pointer]
        self.library.mcblasSetStream.argtypes = [pointer, pointer]
        self.library.mcblasGemmEx.argtypes = [
            pointer,
            integer,
            integer,
            integer,
            integer,
            integer,
            pointer,
            pointer,
            integer,
            integer,
            pointer,
            integer,
            integer,
            pointer,
            pointer,
            integer,
            integer,
            integer,
            integer,
        ]
        self.handle = pointer()
        self.check(self.library.mcblasCreate(ctypes.byref(self.handle)))
        weakref.finalize(self, self.library.mcblasDestroy, self.handle)
        self.alpha, self.beta = ctypes.c_float(1), ctypes.c_float(0)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"bmm_w8a8: mcBLAS returned status {status}")

    def __call__(self, packed, accumulation, m, n, k):
        self.check(
            self.library.mcblasSetStream(
                self.handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            )
        )
        # Column-major BLAS sees B.T @ A.T; BF16 inputs accumulate into FP32.
        self.check(
            self.library.mcblasGemmEx(
                self.handle,
                1,
                0,
                n,
                m,
                k,
                ctypes.byref(self.alpha),
                ctypes.c_void_p(packed.data_ptr() + m * k * 2),
                14,
                k,
                ctypes.c_void_p(packed.data_ptr()),
                14,
                k,
                ctypes.byref(self.beta),
                ctypes.c_void_p(accumulation.data_ptr()),
                0,
                n,
                68,
                -1,
            )
        )


@triton.jit
def e4m3_to_float32(value):
    bits = value.to(tl.int32)
    magnitude = bits & 127
    normal = (magnitude << 20) + (120 << 23)
    subnormal = (magnitude.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True)
    payload = tl.where(magnitude < 8, subnormal, normal)
    payload = tl.where(magnitude == 127, 0x7FC00000, payload)
    return (((bits & 128) << 24) | payload).to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def vector_kernel(
    A,
    B,
    SA,
    SB,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    INTEGER: tl.constexpr,
    E4M3: tl.constexpr = False,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    reduction = tl.arange(0, K)
    a = tl.load(A + row * K + reduction)
    b = tl.load(
        B + cols[:, None] * K + reduction[None, :], cols[:, None] < N, other=0.0
    )
    if E4M3:
        a, b = e4m3_to_float32(a), e4m3_to_float32(b)
    if INTEGER:
        value = tl.sum(a[None, :].to(tl.int32) * b.to(tl.int32), 1).to(tl.float32)
    else:
        value = tl.sum(a[None, :].to(tl.float32) * b.to(tl.float32), 1)
    sa = tl.load(SA + row)
    sb = tl.load(SB + cols, cols < N, other=0.0)
    tl.store(Out + row * N + cols, value * sa * sb, cols < N)


@libentry()
@triton.jit
def int8_matrix_kernel(
    A,
    B,
    SA,
    SB,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    S: tl.constexpr,
):
    BN: tl.constexpr = 64
    BK: tl.constexpr = 128
    pid, split = tl.program_id(0), tl.program_id(1)
    grid_m, grid_n = triton.cdiv(M, BM), triton.cdiv(N, BN)
    first = (pid // (8 * grid_n)) * 8
    size = tl.minimum(grid_m - first, 8)
    pm = first + (pid % (8 * grid_n)) % size
    pn = (pid % (8 * grid_n)) // size
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    reduction = split * (K // S) + tl.arange(0, BK)
    pa = A + (rows[:, None] % M) * K + reduction[None, :]
    pb = B + cols[None, :] * K + reduction[:, None]
    acc = tl.full((BM, BN), 0, tl.int32)
    for _ in range(K // S // BK):
        acc = tl.dot(tl.load(pa), tl.load(pb), acc, out_dtype=tl.int32)
        pa += BK
        pb += BK
    if S == 1:
        sa = tl.load(SA + rows, rows < M, other=0.0)
        sb = tl.load(SB + cols)
        value = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    else:
        value = acc
    tl.store(
        Out + split * M * N + rows[:, None] * N + cols[None, :],
        value,
        rows[:, None] < M,
    )


@libentry()
@triton.jit
def scale_kernel(
    P,
    SA,
    SB,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    INTEGER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INTEGER:
        value = tl.full((BLOCK,), 0, tl.int32)
    else:
        value = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(S):
        value += tl.load(P + split * M * N + offsets, offsets < M * N, other=0.0)
    sa = tl.load(SA + offsets // N, offsets < M * N, other=0.0)
    sb = tl.load(SB + offsets % N, offsets < M * N, other=0.0)
    tl.store(Out + offsets, value.to(tl.float32) * sa * sb, offsets < M * N)


@libentry()
@triton.jit
def fp8_prepare_kernel(
    A, B, H, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    if program < triton.cdiv(M * K, BLOCK):
        value = tl.load(A + offsets, offsets < M * K, other=0.0)
    else:
        value = tl.load(B + offsets - M * K, offsets < (M + N) * K, other=0.0)
    # Both FP8 formats, including subnormals, are exactly representable in BF16.
    tl.store(H + offsets, value.to(tl.bfloat16), offsets < (M + N) * K)


def c550_einsum(A, B, A_scale, B_scale, out):
    _, m, k = A.shape
    n = B.shape[2]
    integer = A.dtype == torch.int8
    if integer and m > 2:
        operator = native_gemm()
        if operator is not None:
            operator(
                [m, n, k],
                A,
                B,
                out,
                A_scale.view(1, m, 1),
                B_scale.view(1, 1, n),
                None,
            )
            return out
    e4m3 = A.dtype == torch.float8_e4m3fn
    if (
        (integer and m <= 4)
        or (e4m3 and m <= 2)
        or (A.dtype == torch.float8_e5m2 and m <= 8)
    ):
        cols = 2 if m == 1 or e4m3 else 4
        vector_a = A.view(torch.uint8) if e4m3 else A
        vector_b = B.view(torch.uint8) if e4m3 else B
        vector_kernel[(m, triton.cdiv(n, cols))](
            vector_a,
            vector_b,
            A_scale,
            B_scale,
            out,
            m,
            n,
            k,
            cols,
            integer,
            e4m3,
            num_warps=4,
        )
        return out
    if integer:
        if m <= 16:
            bm, splits, stages, pipeline = 16, 16, 1, "basic"
        elif m <= 64:
            bm, splits, stages, pipeline = 32, 8, 1, "basic"
        else:
            bm = 64
            splits = 4 if m <= 128 else (2 if m <= 256 else 1)
            stages, pipeline = 2, "cpasync"
        partial = (
            out
            if splits == 1
            else torch.empty((splits, m, n), device=A.device, dtype=torch.int32)
        )
        int8_matrix_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, 64), splits)](
            A,
            B,
            A_scale,
            B_scale,
            partial,
            m,
            n,
            k,
            bm,
            splits,
            num_warps=4,
            num_stages=stages,
            pipeline=pipeline,
        )
        if splits == 1:
            return out
    else:
        device = torch.cuda.current_device()
        if device not in _native_state.blas:
            try:
                _native_state.blas[device] = NativeBlas()
            except OSError:
                # A missing optional native library retains the Triton route.
                _native_state.blas[device] = None
        operator = _native_state.blas[device]
        if operator is None:
            return triton_bmm_w8a8(A, B, A_scale, B_scale, out=out)
        packed = torch.empty((m + n, k), device=A.device, dtype=torch.bfloat16)
        partial = torch.empty((m, n), device=A.device, dtype=torch.float32)
        fp8_prepare_kernel[(triton.cdiv((m + n) * k, 1024),)](
            A, B, packed, m, n, k, 1024
        )
        operator(packed, partial, m, n, k)
        splits = 1
    scale_kernel[(triton.cdiv(m * n, 1024),)](
        partial, A_scale, B_scale, out, m, n, splits, integer, 1024
    )
    return out


def bmm_w8a8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    block_size=(128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    *,
    config=None,
) -> torch.Tensor:
    """Channel-wise batched W8A8 matrix multiplication.

    A has shape (batch, M, K), B has shape (batch, K, N). A_scale is
    (batch, M) or (batch, M, 1); B_scale is (batch, N) or (batch, 1, N).
    INT8 accumulates in INT32; FP8 accumulates in FP32, with exact FP16/BF16
    promotion for matrix multiplication. Scales are applied in FP32 before
    casting to out_dtype.
    block_size is retained for API compatibility and is unused here.
    """
    out = validate_inputs(A, B, A_scale, B_scale, out_dtype, out)
    batch, M, K = A.shape
    N = B.shape[2]
    if batch * M * N == 0:
        return out
    if (
        config is None
        and batch == 1
        and (N, K) == (1024, 4096)
        and (A.dtype != torch.int8 or M <= 512)
        and out_dtype == torch.bfloat16
        and A.is_contiguous()
        and B.stride() == (K * N, 1, K)
        and out.is_contiguous()
        and A_scale.dtype == torch.float32
        and B_scale.dtype == torch.float32
        and A_scale.is_contiguous()
        and B_scale.is_contiguous()
        and "C550" in torch.cuda.get_device_properties(A.device).name
    ):
        with torch.cuda.device(A.device):
            return c550_einsum(A, B, A_scale, B_scale, out)
    if (
        config is None
        and A.dtype == torch.int8
        and out_dtype == torch.bfloat16
        and K > 0
        and K % 32 == 0
        and N % 16 == 0
        and A.is_contiguous()
        and B.stride() == (K * N, 1, K)
        and out.is_contiguous()
        and A_scale.dtype == torch.float32
        and B_scale.dtype == torch.float32
        and A_scale.is_contiguous()
        and B_scale.is_contiguous()
    ):
        with torch.cuda.device(A.device):
            operator = native_gemm()
            if operator is None:
                return triton_bmm_w8a8(
                    A, B, A_scale, B_scale, block_size, out_dtype, out, config=config
                )
            operator(
                [M, N, K],
                A,
                B,
                out,
                A_scale.view(batch, M, 1),
                B_scale.view(batch, 1, N),
                None,
            )
        return out
    return triton_bmm_w8a8(
        A, B, A_scale, B_scale, block_size, out_dtype, out, config=config
    )
