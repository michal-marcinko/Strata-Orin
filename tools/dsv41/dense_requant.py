#!/usr/bin/env python3
"""dense_requant.py - the dense matrices of a DeepSeek-V4.1 pack (attention, compressor, indexer, shared experts; BF16 in the
unsloth GGUF) in a ggml K-quant, through ggml's own quantizer (libggml-base, ctypes), written beside the model as a small file the
pack reads instead of the originals - the experts, the Engram and the embedding/head are not touched, nothing big is copied:

    python tools/dsv41/dense_requant.py --pack PACK --out DIR --type Q5_K [--lib libggml-base.so] [--jobs 8]
    python tools/dsv41/pack_gguf.py --model GGUF_DIR --out PACK2 --dense-override DIR

DIR/dense.bin holds the quantized tensors (4096-aligned), DIR/dense.json {tensor name: [type, file, offset, bytes, shape]}.
Types: Q3_K (3.44 bpw), IQ4_XS (4.25), Q4_K (4.5), Q5_K (5.5), Q6_K (6.56), Q8_0 (8.5).  --include / --exclude are regexes on
the checkpoint name (default: layers.N.attn.* and layers.N.ffn.shared_experts.*, the main layers only, matrices whose rows are
whole 256 blocks).

A pack whose dense tensors are already quantized (the ISTA-DAS IQ1_M sets: Q5_K) has no BF16 to start from: --src-gguf names a
GGUF shard that holds them in BF16 (unsloth's UD-Q8_K_XL 00009-of-00009 holds every dense tensor) and the tensors are taken
from there under the pack's names (pack_gguf.py's map; attn_output_a split into the o_groups wo_a slices), only those the pack
has with the same shape.  --imatrix: llama.cpp's GGUF imatrix (unsloth's imatrix_unsloth.gguf_file: <name>.in_sum2 /
<name>.counts), each row's weights in_sum2 / counts of its matrix (a wo_a slice: its group's, when the imatrix has one a group).
--make-pack NEWPACK then writes a copy of --pack's small files (a few MB) whose tensors.txt reads the DIR's (and any --also
DIR's) tensors instead - the original pack is left as it is; DIR must be on the model's drive:

    python tools/dsv41/dense_requant.py --pack PACK --src-gguf UD-Q8_K_XL-00009-of-00009.gguf --imatrix imatrix_unsloth.gguf_file \\
        --include 'attn\\.wq_b|shared_experts\\.w[13]' --type Q4_K --out MODELS/dense-q4k --make-pack PACKS/pack-q4k
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

GT = {"Q8_0": (8, 32, 34), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210),
      "IQ4_XS": (23, 256, 136)}
_lib = None


def _init(libpath):
    global _lib
    try:
        os.nice(10)
    except OSError:
        pass
    _lib = ctypes.CDLL(libpath)
    _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_void_p]


def _job(j):
    name, path, off, nbytes, rows, cols, tid, qrb, out_path, out_off, imat = j
    im = None
    if imat is not None:
        imat = np.ascontiguousarray(imat, dtype=np.float32)
        im = imat.ctypes.data
    # in chunks of rows (the head's 129,280 x 5,120 in F32 alone would be 2.6 GB), each source range dropped from the page
    # cache once read and each written range once synced (DR_PACE_MBS: at most that many MB/s read - a box whose running
    # engine reads its experts from the same SSD)
    pace = float(os.environ.get("DR_PACE_MBS", "0"))
    sfd, ofd = os.open(path, os.O_RDONLY), os.open(out_path, os.O_WRONLY)
    for r0 in range(0, rows, 8192):
        t0 = time.time()
        n = min(8192, rows - r0)
        so = off + r0 * cols * 2
        raw = np.frombuffer(os.pread(sfd, n * cols * 2, so), dtype=np.uint16)
        x = np.ascontiguousarray((raw.astype(np.uint32) << 16).view(np.float32).reshape(n, cols))
        dst = np.empty(n * qrb, dtype=np.uint8)
        got = _lib.ggml_quantize_chunk(tid, x.ctypes.data, dst.ctypes.data, 0, n, cols, im)
        if got != n * qrb:
            raise RuntimeError("%s: ggml_quantize_chunk wrote %d of %d bytes" % (name, got, n * qrb))
        # (the result is checked against the source and the pack's own tensor by dense_check.py, not here)
        os.pwrite(ofd, dst.tobytes(), out_off + r0 * qrb)
        os.posix_fadvise(sfd, so, n * cols * 2, os.POSIX_FADV_DONTNEED)
        os.fdatasync(ofd)
        os.posix_fadvise(ofd, out_off + r0 * qrb, n * qrb, os.POSIX_FADV_DONTNEED)
        if pace > 0:
            time.sleep(max(0.0, n * cols * 2 / (pace * 1e6) - (time.time() - t0)))
    os.close(sfd)
    os.close(ofd)
    return name, rows * qrb


def read_pack(pack):
    files, tensors, cfg = {}, [], {}
    for line in open(os.path.join(pack, "tensors.txt"), encoding="utf-8"):
        f = line.split()
        if f and f[0] == "file":
            files[int(f[1])] = line.split(None, 2)[2].strip()
        elif f and f[0] == "t":
            tensors.append((f[1], f[2], int(f[3]), int(f[4]), int(f[5]), [int(x) for x in f[6:]]))
    for line in open(os.path.join(pack, "ds4.txt"), encoding="utf-8"):
        k, _, v = line.strip().partition(" ")
        cfg[k] = v
    return files, tensors, cfg


def gguf_bf16(src, o_groups):
    """{pack name: (abs offset, bytes, [rows, cols], gguf name, group or -1)} of the BF16 matrices in a GGUF shard."""
    import pack_gguf as PG
    g = PG.Gguf(pathlib.Path(src))
    out = {}
    for name, typ, dims, off in g.tensors:
        if typ != 30:   # BF16
            continue
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if m and m.group(2) in PG.LAYER:
            ck = "layers.%s.%s" % (m.group(1), PG.LAYER[m.group(2)])
        elif name in PG.GLOBAL:
            ck = PG.GLOBAL[name]
        else:
            continue
        shape = list(reversed(dims))
        nel = int(np.prod(shape))
        base = g.data_start + off
        if ck.endswith(".attn.wo_a.slice"):   # [o_groups * o_lora][in] (or [o_groups][o_lora][in]) -> one entry a group
            if len(shape) == 3:
                shape = [shape[0] * shape[1], shape[2]]
            part = nel * 2 // o_groups
            for i in range(o_groups):
                out["%s.%d.weight" % (ck, i)] = (base + i * part, part, [shape[0] // o_groups, shape[1]], name, i)
            continue
        if len(shape) == 2:
            out[ck] = (base, nel * 2, shape, name, -1)
    return out


def read_imatrix(path):
    """llama.cpp's GGUF imatrix: {gguf tensor name: in_sum2 / counts as float32 [n_mat][cols]}."""
    import pack_gguf as PG
    g = PG.Gguf(pathlib.Path(path))
    t = {name: (dims, off) for name, typ, dims, off in g.tensors if typ == 0}
    out = {}
    for name, (dims, off) in t.items():
        if not name.endswith(".in_sum2") or name[:-8] + ".counts" not in t or "_exps." in name:   # (the routed experts' ~770 MB: not dense)
            continue
        s = np.fromfile(path, dtype=np.float32, count=int(np.prod(dims)), offset=g.data_start + off).reshape(-1, dims[0])
        cd, co = t[name[:-8] + ".counts"]
        c = np.fromfile(path, dtype=np.float32, count=int(np.prod(cd)), offset=g.data_start + co).reshape(-1)
        out[name[:-8]] = s / np.maximum(c, 1.0)[:, None]
    return out


def make_pack(pack, dirs, newpack):
    files, tensors, cfg = read_pack(pack)
    model_dir = cfg["model_dir"]
    os.makedirs(newpack, exist_ok=True)
    for f in os.listdir(pack):
        p = os.path.join(pack, f)
        if f != "tensors.txt":
            (shutil.copytree(p, os.path.join(newpack, f), dirs_exist_ok=True) if os.path.isdir(p) else shutil.copy2(p, newpack))
    lines = open(os.path.join(pack, "tensors.txt"), encoding="utf-8").read().splitlines()
    repl, nfi = {}, max(files) + 1
    new_files = []
    for d in dirs:
        dj = json.load(open(os.path.join(d, "dense.json")))
        rel = os.path.relpath(os.path.join(os.path.abspath(d), "dense.bin"), model_dir)
        new_files.append("file %d %s" % (nfi, rel))
        for name, (tname, _, off, nb, shape) in dj.items():
            repl[name] = "t %s %s %d %d %d %s" % (name, tname, nfi, off, nb, " ".join(map(str, shape)))
        nfi += 1
    last_file = max(i for i, l in enumerate(lines) if l.startswith("file "))
    out, seen = [], set()
    for i, l in enumerate(lines):
        f = l.split(None, 2)
        if f and f[0] == "t" and f[1] in repl:
            out.append(repl[f[1]])
            seen.add(f[1])
        else:
            out.append(l)
        if i == last_file:
            out += new_files
    missing = set(repl) - seen
    if missing:
        raise SystemExit("not in the pack: %s" % ", ".join(sorted(missing)[:8]))
    open(os.path.join(newpack, "tensors.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("pack %s: %d tensors from %s" % (newpack, len(seen), ", ".join(dirs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", choices=sorted(GT), help="(left out with --make-pack: only the pack is written, from --out's dense.json)")
    ap.add_argument("--lib", default="/data/dsv41/build-hip/bin/libggml-base.so")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--include", default=r"^layers\.\d+\.(attn\.|ffn\.shared_experts\.)")
    ap.add_argument("--exclude", default=r"(norm|sink|gate\.bias|\.gate\.weight$|hc_|weights_proj)")
    ap.add_argument("--src-gguf", help="a GGUF shard holding the dense tensors in BF16 (the pack's own are quantized already)")
    ap.add_argument("--imatrix", help="llama.cpp's GGUF imatrix (in_sum2 / counts) for the --src-gguf tensors")
    ap.add_argument("--make-pack", help="write a copy of --pack (small files only) that reads --out's tensors instead")
    ap.add_argument("--also", action="append", default=[], help="(--make-pack) another dense_requant.py folder to read too")
    a = ap.parse_args()
    if a.type is None:
        if not a.make_pack:
            raise SystemExit("--type or --make-pack")
        make_pack(a.pack, [a.out] + a.also, a.make_pack)
        return
    tid, per, bpb = GT[a.type]
    files, tensors, cfg = read_pack(a.pack)
    model_dir = cfg["model_dir"]
    inc, exc = re.compile(a.include), re.compile(a.exclude)
    imat = read_imatrix(a.imatrix) if a.imatrix else {}
    if a.src_gguf:
        src = gguf_bf16(a.src_gguf, int(cfg.get("o_groups", 8)))
        sel = []   # (name, path, off, bytes, shape, imatrix row)
        for name, dt, fi, o, nb, shape in tensors:
            if name not in src or not inc.search(name) or exc.search(name):
                continue
            so, sb, sshape, gname, grp = src[name]
            if sshape != shape or shape[1] % per:
                raise SystemExit("%s: %s in the pack, %s in %s" % (name, shape, sshape, a.src_gguf))
            im = imat.get(gname)
            if a.imatrix and im is None:
                raise SystemExit("%s (%s): not in the imatrix" % (name, gname))
            if im is not None:
                im = im[grp if grp >= 0 and im.shape[0] > 1 else 0]
                if grp >= 0 and im.shape[0] == shape[1] * int(cfg.get("o_groups", 8)):   # (llama.cpp's grouped mul_mat: the
                    im = im[grp * shape[1]:(grp + 1) * shape[1]]                          #  groups' inputs one after another)
                if im.shape[0] != shape[1]:
                    raise SystemExit("%s: imatrix of %d columns, the matrix has %d" % (name, im.shape[0], shape[1]))
            sel.append((name, a.src_gguf, so, sb, shape, im))
    else:
        sel = [(t[0], os.path.join(model_dir, files[t[2]]), t[3], t[4], t[5], None) for t in tensors
               if t[1] == "BF16" and len(t[5]) == 2 and t[5][1] % per == 0 and inc.search(t[0]) and not exc.search(t[0])]
    if not sel:
        raise SystemExit("no tensors selected")
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "dense.bin")
    jobs, idx, off = [], {}, 0
    for name, path, o, nb, shape, im in sel:
        rows, cols = shape
        qrb = cols // per * bpb
        jobs.append((name, path, o, nb, rows, cols, tid, qrb, out_path, off, im))
        idx[name] = [a.type, "dense.bin", off, rows * qrb, shape]
        off += (rows * qrb + 4095) // 4096 * 4096
    with open(out_path, "wb") as f:
        f.truncate(off)
    src_b = sum(t[3] for t in sel)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.lib,)) as pool:
        for i, (name, n) in enumerate(pool.imap_unordered(_job, jobs)):
            if i % 40 == 0:
                print("%d/%d %s %.0f s" % (i, len(jobs), name, time.time() - t0), flush=True)
    json.dump(idx, open(os.path.join(a.out, "dense.json"), "w"))
    print("dense %s: %d tensors, %.2f GB BF16 -> %.2f GB, %.0f s%s" % (a.type, len(sel), src_b / 1e9, sum(v[3] for v in idx.values()) / 1e9,
                                                                  time.time() - t0, " (imatrix)" if a.imatrix else ""))
    if a.make_pack:
        make_pack(a.pack, [a.out] + a.also, a.make_pack)


if __name__ == "__main__":
    main()
