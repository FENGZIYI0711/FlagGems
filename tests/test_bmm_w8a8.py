import json
from pathlib import Path

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 17, 37, 65),
        (3, 64, 80, 128),
        (2, 11, 13, 0),
        (1, 0, 13, 19),
        (1, 31, 47, 257),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.bmm_w8a8
def test_bmm_w8a8(dtype, out_dtype, shape, transpose):
    if (
        dtype == torch.float8_e4m3fn
        and flag_gems.vendor_name == "nvidia"
        and torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip("Triton FP8 E4M3FN conversion requires NVIDIA SM89 or newer")
    torch.manual_seed(20261019)
    batch, m, n, k = shape
    device = flag_gems.device
    b_shape = (batch, n, k) if transpose else (batch, k, n)
    if dtype == torch.int8:
        a = torch.randint(-128, 128, (batch, m, k), device=device, dtype=dtype)
        b = torch.randint(-128, 128, b_shape, device=device, dtype=dtype)
    else:
        a = (torch.randn(batch, m, k, device=device) * 3).to(dtype)
        b = (torch.randn(b_shape, device=device) * 3).to(dtype)
    if transpose:
        b = b.transpose(1, 2)
    sa = (torch.rand(batch, m * 2, device=device) * 0.005)[:, ::2]
    sb = (torch.rand(batch, n * 2, device=device) * 0.005)[:, ::2]
    out = torch.empty(batch, n, m, device=device, dtype=out_dtype).transpose(1, 2)
    expected = (torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]).to(
        out_dtype
    )
    actual = flag_gems.bmm_w8a8(
        a, b, sa[:, :, None], sb[:, None, :], out_dtype=out_dtype, out=out
    )
    assert actual is out
    torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=0.002 if out_dtype == torch.float16 else 0.01
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 97, 80, 128),
        (2, 64, 80, 128),
        (3, 129, 256, 256),
        (1, 257, 128, 7168),
        (2, 64, 32, 32),
        (1, 163, 2112, 1024),
    ],
)
@pytest.mark.bmm_w8a8
def test_bmm_w8a8_aligned(shape):
    torch.manual_seed(20261020)
    batch, m, n, k = shape
    a = torch.randint(
        -128, 128, (batch, m, k), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (batch, n, k), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(batch, m, device=flag_gems.device) * 0.001
    sb = torch.rand(batch, n, device=flag_gems.device) * 0.001
    expected = (
        torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
    ).bfloat16()
    actual = flag_gems.bmm_w8a8(a, b, sa, sb)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.01)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX graph fast path")
@pytest.mark.bmm_w8a8
def test_bmm_w8a8_graph_replay():
    torch.manual_seed(20261020)
    a = torch.randint(
        -128, 128, (2, 64, 128), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (2, 80, 128), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(2, 64, device=flag_gems.device) * 0.001
    sb = torch.rand(2, 80, device=flag_gems.device) * 0.001
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.bmm_w8a8(a, b, sa, sb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = flag_gems.bmm_w8a8(a, b, sa, sb)
    for _ in range(3):
        a.random_(-128, 128)
        sa.uniform_(0, 0.001)
        sb.uniform_(0, 0.001)
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
        ).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.01)


# FlagOSTune f4f65a7d779a0df3b017e9a80b75a7b9d651c639, DeepSeek-V4-Flash.
# Reuse contraction dimensions, with channel-wise rather than block-wise scales.
EINSUM_SHAPES = [
    tuple(row["shape"])
    for row in json.loads(
        (Path(__file__).parent / "data/bmm_w8a8_einsum_shapes.json").read_text()
    )
]


@pytest.mark.bmm_w8a8
@pytest.mark.parametrize("shape", EINSUM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
def test_bmm_w8a8_einsum_shapes(shape, dtype, monkeypatch):
    if (
        dtype == torch.float8_e4m3fn
        and flag_gems.vendor_name == "nvidia"
        and torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip("Triton FP8 E4M3FN conversion requires NVIDIA SM89 or newer")
    torch.manual_seed(20261027)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    b, h, d, r = shape
    device = flag_gems.device
    # bhr,hdr->bhd becomes (h,b,r) @ (h,r,d), followed by a view transpose.
    x = (torch.randn(b, h, r, device=device) * 3).to(dtype)
    y = (torch.randn(h, d, r, device=device) * 3).to(dtype)
    xs = torch.rand(b, h, device=device) * 0.01
    ys = torch.rand(h, d, device=device) * 0.01
    expected = (
        torch.einsum("bhr,hdr->bhd", x.float(), y.float())
        * xs[:, :, None]
        * ys[None, :, :]
    ).bfloat16()
    actual = flag_gems.bmm_w8a8(
        x.permute(1, 0, 2), y.transpose(1, 2), xs.transpose(0, 1), ys
    ).transpose(0, 1)
    assert actual.shape == (b, h, d)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX graph fast paths")
@pytest.mark.bmm_w8a8
@pytest.mark.parametrize("m", [1, 4, 16, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
def test_bmm_w8a8_einsum_graph_full_range(m, dtype):
    torch.manual_seed(20261027)
    n, k = 1024, 4096
    device = flag_gems.device
    if dtype == torch.int8:
        codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    else:
        codes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        codes = codes[torch.isfinite(codes.view(dtype).float())]
    codes = codes.to(device)

    def sample(shape):
        return codes[torch.randint(codes.numel(), shape, device=device)].view(dtype)

    a = sample((1, m, k))
    b = sample((1, n, k)).transpose(1, 2)
    scale_limit = 1e-6 if dtype == torch.float8_e5m2 else 1e-3
    sa = torch.rand(1, m, device=device) * scale_limit
    sb = torch.rand(1, n, device=device) * scale_limit
    out = torch.empty((1, m, n), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actual = flag_gems.bmm_w8a8(a, b, sa, sb, out=out)
    assert actual is out
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flag_gems.bmm_w8a8(a, b, sa, sb, out=out)
    for _ in range(2):
        a.copy_(sample(a.shape))
        b.copy_(sample((1, n, k)).transpose(1, 2))
        sa.uniform_(0, scale_limit)
        sb.uniform_(0, scale_limit)
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
        ).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.01)
