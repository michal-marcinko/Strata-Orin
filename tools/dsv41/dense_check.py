#!/usr/bin/env python3
"""dense_check.py - is a pack's quantized dense tensor the stock one, and does a dense_requant.py override match it?

    python tools/dsv41/dense_check.py --pack PACK --src-gguf UD-Q8_K_XL-00009-of-00009.gguf --names 'layers\\.(0|20|39)\\.attn\\.wq_b' [--dense DIR] [--rank]

Per tensor (W = the BF16 source, P = the pack's tensor dequantized, N = DIR's, dequantized; gguf-py's numpy dequantizers):
  rel(P-W)   the pack's error against the stock weights, ||P - W|| / ||W||
  rel(N-W)   the override's error against them (with --dense)
  corr(N,P)  the override against the pack's tensor (a wrong layout or another matrix shows here)
  s1/s2      (--rank) the top two singular values' ratio of P - W (randomized, 24 vectors): quantization error is spread over many
             directions (about 1), a weight edit made at full precision before quantizing (the uncensored sets' one-direction
             refusal edit, W - a r r^T W) is one direction far above the rest
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for p in (HERE.parent.parent / "third_party" / "llama.cpp" / "gguf-py",):
    sys.path.insert(0, str(p))
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import dense_requant as DR  # noqa: E402

QT = {"Q3_K": Q.Q3_K, "Q4_K": Q.Q4_K, "Q5_K": Q.Q5_K, "Q6_K": Q.Q6_K, "Q8_0": Q.Q8_0, "IQ4_XS": Q.IQ4_XS}
CH = 4096   # rows a chunk


PACE = float(os.environ.get("DR_PACE_MBS", "0"))   # (at most that many MB/s read: a box whose engine reads the same SSD)


def read(path, off, n):   # n bytes at off, dropped from the page cache after (the running engine's experts stay cached)
    t0 = time.time()
    fd = os.open(path, os.O_RDONLY)
    b = os.pread(fd, n, off)
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    if PACE > 0:
        time.sleep(max(0.0, n / (PACE * 1e6) - (time.time() - t0)))
    return b


def rows_q(path, off, nbytes, shape, tname, r0, r1):
    rb = nbytes // shape[0]
    raw = np.frombuffer(read(path, off + r0 * rb, (r1 - r0) * rb), dtype=np.uint8).reshape(r1 - r0, rb)
    return dequantize(raw, QT[tname]).astype(np.float32).reshape(r1 - r0, shape[1])


def rows_bf16(path, off, shape, r0, r1):
    raw = np.frombuffer(read(path, off + r0 * shape[1] * 2, (r1 - r0) * shape[1] * 2), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(r1 - r0, shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--src-gguf", required=True)
    ap.add_argument("--names", required=True, help="regex on the pack's tensor names")
    ap.add_argument("--dense", help="a dense_requant.py folder to check against the pack's tensors")
    ap.add_argument("--rank", action="store_true")
    a = ap.parse_args()
    files, tensors, cfg = DR.read_pack(a.pack)
    md = cfg["model_dir"]
    src = DR.gguf_bf16(a.src_gguf, int(cfg.get("o_groups", 8)))
    dj = json.load(open(os.path.join(a.dense, "dense.json"))) if a.dense else {}
    rx = re.compile(a.names)
    print("%-44s %-6s %-6s %9s %9s %10s %8s" % ("tensor", "pack", "new", "rel(P-W)", "rel(N-W)", "corr(N,P)", "s1/s2"))
    for name, dt, fi, off, nb, shape in tensors:
        if not rx.search(name) or name not in src or dt not in QT:
            continue
        so, sb, sshape, _, _ = src[name]
        if sshape != shape:
            print("%s: shape %s vs source %s" % (name, shape, sshape))
            continue
        ppath = os.path.join(md, files[fi])
        nd = dj.get(name)
        rows = shape[0]
        sw = sd = sn = spn = sp2 = sn2 = 0.0
        rng = np.random.default_rng(0)
        om = rng.standard_normal((shape[1], 24)).astype(np.float32) if a.rank else None
        ys = []
        for r0 in range(0, rows, CH):
            r1 = min(rows, r0 + CH)
            W = rows_bf16(a.src_gguf, so, shape, r0, r1)
            P = rows_q(ppath, off, nb, shape, dt, r0, r1)
            D = P - W
            sw += float(np.square(W, dtype=np.float64).sum())
            sd += float(np.square(D, dtype=np.float64).sum())
            if nd is not None:
                N = rows_q(os.path.join(a.dense, nd[1]), nd[2], nd[3], nd[4], nd[0], r0, r1)
                sn += float(np.square(N - W, dtype=np.float64).sum())
                spn += float((P.astype(np.float64) * N).sum())
                sp2 += float(np.square(P, dtype=np.float64).sum())
                sn2 += float(np.square(N, dtype=np.float64).sum())
            if a.rank:
                ys.append(D @ om)
        s12 = float("nan")
        if a.rank:   # randomized range finder: Y = D Om, Q = orth(Y), B = Q^T D, singular values of B
            Qm, _ = np.linalg.qr(np.concatenate(ys))
            B = np.zeros((Qm.shape[1], shape[1]), dtype=np.float64)
            for r0 in range(0, rows, CH):
                r1 = min(rows, r0 + CH)
                D = rows_q(ppath, off, nb, shape, dt, r0, r1) - rows_bf16(a.src_gguf, so, shape, r0, r1)
                B += Qm[r0:r1].T.astype(np.float64) @ D
            s = np.linalg.svd(B, compute_uv=False)
            s12 = s[0] / s[1]
        rn = (sn / sw) ** 0.5 if nd is not None else float("nan")
        cor = spn / (sp2 * sn2) ** 0.5 if nd is not None else float("nan")
        print("%-44s %-6s %-6s %9.5f %9.5f %10.6f %8.2f" % (name, dt, nd[0] if nd else "-", (sd / sw) ** 0.5, rn, cor, s12), flush=True)


if __name__ == "__main__":
    main()
