import gc
import json
import os
import statistics
from pathlib import Path

import torch

from flag_gems import bmm_w8a8


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        o = fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    return g, o


def measure(gs):
    values = [[] for g in gs]
    for i in range(31):
        for j in (range(len(gs)) if i % 2 else reversed(range(len(gs)))):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            for _ in range(5):
                gs[j].replay()
            b.record()
            b.synchronize()
            values[j].append(a.elapsed_time(b) * 200)
    return values


def main():
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    dtypes = (
        (torch.int8,)
        if os.environ.get("BMM_INT8_ONLY") == "1"
        else (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)
    )
    if os.environ.get("BMM_EINSUM_SHAPES") == "1":
        source = (
            Path(__file__).resolve().parents[1]
            / "tests/data/bmm_w8a8_einsum_shapes.json"
        )
        records = json.loads(source.read_text())
        # Source labels are b,h,d,r; BMM dimensions are batch=h, M=b, N=d, K=r.
        shapes = [
            (
                row["shape"][1],
                row["shape"][0],
                row["shape"][2],
                row["shape"][3],
                row["count"],
            )
            for row in records
        ]
        shape_source = "fp8_einsum__int8_einsum"
    else:
        shapes = [
            (1, 16384, 7168, 1024, 12),
            (1, 16384, 2112, 7168, 5),
            (1, 448, 7168, 256, 4),
            (1, 14429, 7168, 1024, 4),
            (1, 14429, 2112, 7168, 1),
        ]
        shape_source = "bmm_w8a8_fp8__int8"
    for dt in dtypes:
        for bs, m, n, k, count in shapes:
            raw_a = torch.randn(bs, m, k, device="cuda", dtype=torch.bfloat16)
            raw_b = torch.randn(
                bs, n, k, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            limit = (
                127
                if dt == torch.int8
                else (448 if dt == torch.float8_e4m3fn else 57344)
            )
            sa = raw_a.float().abs().amax(2) / limit
            sb = raw_b.float().abs().amax(1) / limit
            aq = raw_a.float() / sa[:, :, None]
            bq = raw_b.float() / sb[:, None, :]
            if dt == torch.int8:
                aq = aq.round()
                bq = bq.round()
            a = aq.to(dt)
            b = bq.transpose(1, 2).contiguous().to(dt).transpose(1, 2)
            ref = (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            f = lambda a=a, b=b, sa=sa, sb=sb: bmm_w8a8(a, b, sa, sb)
            base = lambda a=a, b=b, sa=sa, sb=sb: (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            # Raw BF16 BMM is a separate lower-work baseline, not numerically equivalent.
            ab = raw_a
            bb = raw_b
            raw = lambda ab=ab, bb=bb: torch.bmm(ab, bb)
            out = f()
            torch.testing.assert_close(out, ref, atol=0.002, rtol=0.01)
            gs = []
            outs = []
            for fn in (f, base, raw):
                g, o = capture(fn)
                gs.append(g)
                outs.append(o)
            samples = measure(gs)
            us = [statistics.median(x) for x in samples]
            rows.append(
                dict(
                    dtype=str(dt),
                    shape=[bs, m, n, k],
                    batch_assumed=shape_source == "bmm_w8a8_fp8__int8",
                    shape_source=shape_source,
                    count=count,
                    candidate_us=us[0],
                    torch_channel_us=us[1],
                    raw_bf16_us=us[2],
                    speedup=us[1] / us[0],
                    raw_speedup=us[2] / us[0],
                    samples=samples,
                    correct=True,
                )
            )
            json.dump(
                rows, open(os.environ.get("BMM_RESULT", "results.json"), "w"), indent=2
            )
            print(
                rows[-1]["dtype"],
                [bs, m, n, k],
                us,
                "speedup",
                us[1] / us[0],
                "raw",
                us[2] / us[0],
                flush=True,
            )
            del gs, outs, g, o, out, ref, a, b, ab, bb, raw_a, raw_b, aq, bq
            gc.collect()
            torch.cuda.empty_cache()
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
