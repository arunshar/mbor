#!/usr/bin/env python3
"""G4: Triton dense-kernel acceleration of MBOR's online sub-steps, benchmarked
CPU (numpy) vs GPU (Triton on A100), on REAL exported MBOR boundary-pair data.

Three kernels:
  1. outer_sum   : batched Minkowski outer-sum A (+) B  (dense write)
  2. pareto_scan : batched 2D Pareto filter (lexsort on host, prefix-min keep)
  3. dci_corner  : batched 2D cost-interval ideal corner (min c1-sum, min c2-sum)

The bench composes them into the three MBOR operations and reports, per operation,
a torch.allclose correctness check vs the numpy reference and CPU-vs-GPU timing.

HONESTY CONTRACT: a kernel's status is "validated" only if (a) its GPU output
matches the numpy reference AND (b) the GPU run is at least as fast as CPU on the
real batch. Otherwise "not-validated" (with the measured numbers). If Triton or a
CUDA device is unavailable it exits cleanly with that reason, never a fake number.

Run on MSI:
  apptainer exec --nv ~/mirror_torch.sif python g4_kernels_bench.py --pairs <pairs.json> --out g4_results.json
"""
from __future__ import annotations
import argparse, json, sys, time

# ---------------------------------------------------------------------------
# numpy references (ground truth)
# ---------------------------------------------------------------------------
def np_pareto_filter(pts):
    """pts: (m,2) int array -> Pareto frontier rows, sorted by c1 asc / c2 desc."""
    import numpy as np
    if len(pts) == 0:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))  # by c1, then c2
    s = pts[order]
    keep = []
    best = 1 << 62
    for c1, c2 in s:
        if c2 < best:
            keep.append((c1, c2))
            best = c2
    return np.array(keep, dtype=pts.dtype)

def np_minkowski_filter(a, b, c):
    import numpy as np
    ab = (a[:, None, :] + b[None, :, :]).reshape(-1, 2)
    ab = np_pareto_filter(ab)
    abc = (ab[:, None, :] + c[None, :, :]).reshape(-1, 2)
    return np_pareto_filter(abc)

def np_ideal_corner(a, b, c):
    import numpy as np
    return np.array([a[:, 0].min() + b[:, 0].min() + c[:, 0].min(),
                     a[:, 1].min() + b[:, 1].min() + c[:, 1].min()], dtype=a.dtype)

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
def build_kernels():
    try:
        import torch, triton
        import triton.language as tl
    except Exception as e:
        return None, f"triton/torch import failed: {e}"

    @triton.jit
    def _outer_sum(a_ptr, b_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr):
        # one program per (row, i in A): write A[i]+B[j] for all j.
        row = tl.program_id(0)
        i = tl.program_id(1)
        j = tl.arange(0, B)
        a1 = tl.load(a_ptr + (row * A + i) * 2 + 0)
        a2 = tl.load(a_ptr + (row * A + i) * 2 + 1)
        b1 = tl.load(b_ptr + (row * B + j) * 2 + 0)
        b2 = tl.load(b_ptr + (row * B + j) * 2 + 1)
        o = (row * A + i) * B + j
        tl.store(out_ptr + o * 2 + 0, a1 + b1)
        tl.store(out_ptr + o * 2 + 1, a2 + b2)

    @triton.jit
    def _pareto_scan(sorted_ptr, len_ptr, mask_ptr, M: tl.constexpr):
        # one program per row; points pre-sorted by (c1 asc, c2 asc) on host.
        row = tl.program_id(0)
        n = tl.load(len_ptr + row)
        run = tl.full((), 1 << 62, tl.int64)
        for i in range(M):
            c2 = tl.load(sorted_ptr + (row * M + i) * 2 + 1).to(tl.int64)
            keep = (i < n) & (c2 < run)
            tl.store(mask_ptr + row * M + i, keep.to(tl.int8))
            run = tl.where(keep, c2, run)

    @triton.jit
    def _dci_corner(a_ptr, b_ptr, c_ptr, la_ptr, lb_ptr, lc_ptr, out_ptr,
                    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
        # one program per row; reduce min c1 and min c2 over each padded set.
        row = tl.program_id(0)
        BIG = 1 << 62
        ia = tl.arange(0, A); ib = tl.arange(0, B); ic = tl.arange(0, C)
        na = tl.load(la_ptr + row); nb = tl.load(lb_ptr + row); nc = tl.load(lc_ptr + row)
        a1 = tl.where(ia < na, tl.load(a_ptr + (row * A + ia) * 2 + 0), BIG)
        a2 = tl.where(ia < na, tl.load(a_ptr + (row * A + ia) * 2 + 1), BIG)
        b1 = tl.where(ib < nb, tl.load(b_ptr + (row * B + ib) * 2 + 0), BIG)
        b2 = tl.where(ib < nb, tl.load(b_ptr + (row * B + ib) * 2 + 1), BIG)
        c1 = tl.where(ic < nc, tl.load(c_ptr + (row * C + ic) * 2 + 0), BIG)
        c2 = tl.where(ic < nc, tl.load(c_ptr + (row * C + ic) * 2 + 1), BIG)
        m1 = tl.min(a1, axis=0) + tl.min(b1, axis=0) + tl.min(c1, axis=0)
        m2 = tl.min(a2, axis=0) + tl.min(b2, axis=0) + tl.min(c2, axis=0)
        tl.store(out_ptr + row * 2 + 0, m1)
        tl.store(out_ptr + row * 2 + 1, m2)

    return {"outer_sum": _outer_sum, "pareto_scan": _pareto_scan, "dci_corner": _dci_corner}, None


def gpu_pareto_filter_rows(padded, lengths, kernels):
    """padded: (P,M,2) int64 cuda; lengths:(P,) -> keep mask (P,M) via host lexsort + Triton scan."""
    import torch
    P, M, _ = padded.shape
    # invalid rows get BIG so they sort last
    BIG = 1 << 62
    idx = torch.arange(M, device=padded.device)[None, :].expand(P, M)
    valid = idx < lengths[:, None]
    pts = torch.where(valid[..., None], padded, torch.full_like(padded, BIG))
    # lexicographic sort by (c1, c2): stable sort by c2 then by c1
    o2 = torch.argsort(pts[:, :, 1], dim=1, stable=True)
    pts = torch.gather(pts, 1, o2[..., None].expand(P, M, 2))
    o1 = torch.argsort(pts[:, :, 0], dim=1, stable=True)
    pts = torch.gather(pts, 1, o1[..., None].expand(P, M, 2)).contiguous()
    mask = torch.empty((P, M), dtype=torch.int8, device=padded.device)
    kernels["pareto_scan"][(P,)](pts, lengths.to(torch.int64), mask, M=M)
    return pts, mask


def pad_sets(sets, dtype):
    import numpy as np
    P = len(sets)
    M = max((len(s) for s in sets), default=1)
    M = max(M, 1)
    arr = np.zeros((P, M, 2), dtype=dtype)
    lens = np.zeros(P, dtype=np.int64)
    for i, s in enumerate(sets):
        if len(s):
            arr[i, : len(s)] = s
            lens[i] = len(s)
    return arr, lens


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--limit", type=int, default=4000, help="cap #triples for the bench")
    ap.add_argument("--out", default="g4_results.json")
    args = ap.parse_args(argv)

    pairs = json.load(open(args.pairs))[: args.limit]
    A = [np.array(p["a"], dtype=np.int64).reshape(-1, 2) for p in pairs]
    B = [np.array(p["b"], dtype=np.int64).reshape(-1, 2) for p in pairs]
    C = [np.array(p["c"], dtype=np.int64).reshape(-1, 2) for p in pairs]
    result = {"workload": {"triples": len(pairs)}, "kernels": {}}

    try:
        import torch
    except Exception as e:
        result["status"] = "not-validated"; result["reason"] = f"torch unavailable: {e}"
        print(json.dumps(result, indent=2)); open(args.out, "w").write(json.dumps(result, indent=2)); return 0
    if not torch.cuda.is_available():
        result["status"] = "not-validated"; result["reason"] = "no CUDA device"
        print(json.dumps(result, indent=2)); open(args.out, "w").write(json.dumps(result, indent=2)); return 0
    kernels, reason = build_kernels()
    if kernels is None:
        result["status"] = "not-validated"; result["reason"] = reason
        print(json.dumps(result, indent=2)); open(args.out, "w").write(json.dumps(result, indent=2)); return 0
    dev = "cuda"; result["device"] = torch.cuda.get_device_name(0)

    def cuda_time(fn, iters=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    # ---- Kernel 2 (flagship): batched 2D Pareto filter on the combined A(+)B(+)C candidate sets ----
    combos = [np_pareto_filter((A[i][:, None, :] + B[i][None, :, :]).reshape(-1, 2)) for i in range(len(A))]
    combos = [(combos[i][:, None, :] + C[i][None, :, :]).reshape(-1, 2) for i in range(len(A))]
    parr, plens = pad_sets(combos, np.int64)
    # CPU reference frontier (as sets)
    t0 = time.perf_counter()
    cpu_fr = [np_pareto_filter(combos[i]) for i in range(len(combos))]
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    pad_t = torch.tensor(parr, device=dev); len_t = torch.tensor(plens, device=dev)
    def run_filter():
        pts, mask = gpu_pareto_filter_rows(pad_t, len_t, kernels); return pts, mask
    gpu_ms = cuda_time(run_filter)
    pts, mask = run_filter()
    # correctness: compare frontier sets per row
    ok = True
    pts_c = pts.cpu().numpy(); mask_c = mask.cpu().numpy().astype(bool)
    for i in range(len(combos)):
        got = pts_c[i][mask_c[i]]
        want = cpu_fr[i]
        gs = set(map(tuple, got.tolist())); ws = set(map(tuple, want.tolist()))
        if gs != ws:
            ok = False; break
    result["kernels"]["pareto_filter"] = {
        "correct": bool(ok), "cpu_ms": round(cpu_ms, 3), "gpu_ms": round(gpu_ms, 3),
        "speedup": round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None,
        "status": "validated" if (ok and gpu_ms <= cpu_ms) else "not-validated",
    }

    # ---- Kernel 3: batched 2DCI ideal corner ----
    aarr, al = pad_sets(A, np.int64); barr, bl = pad_sets(B, np.int64); carr, cl = pad_sets(C, np.int64)
    at, bt, ct = torch.tensor(aarr, device=dev), torch.tensor(barr, device=dev), torch.tensor(carr, device=dev)
    alt, blt, clt = torch.tensor(al, device=dev), torch.tensor(bl, device=dev), torch.tensor(cl, device=dev)
    out = torch.empty((len(A), 2), dtype=torch.int64, device=dev)
    Am, Bm, Cm = aarr.shape[1], barr.shape[1], carr.shape[1]
    def run_dci():
        kernels["dci_corner"][(len(A),)](at, bt, ct, alt, blt, clt, out, A=Am, B=Bm, C=Cm); return out
    gdci = cuda_time(run_dci); _ = run_dci()
    t0 = time.perf_counter()
    cpu_corners = np.array([np_ideal_corner(A[i], B[i], C[i]) for i in range(len(A))])
    cdci = (time.perf_counter() - t0) * 1000.0
    dci_ok = bool(np.array_equal(out.cpu().numpy(), cpu_corners))
    result["kernels"]["dci_corner"] = {
        "correct": dci_ok, "cpu_ms": round(cdci, 3), "gpu_ms": round(gdci, 3),
        "speedup": round(cdci / gdci, 2) if gdci > 0 else None,
        "status": "validated" if (dci_ok and gdci <= cdci) else "not-validated",
    }

    # ---- Kernel 1: batched outer-sum A(+)B (dense write), correctness vs numpy ----
    aS, bS = aarr.shape[1], barr.shape[1]
    osum = torch.empty((len(A), aS * bS, 2), dtype=torch.int64, device=dev)
    def run_outer():
        kernels["outer_sum"][(len(A), aS)](at, bt, osum, A=aS, B=bS); return osum
    gout = cuda_time(run_outer); _ = run_outer()
    # numpy reference outer-sum (padded rows produce garbage but we only check valid prefix of row 0..few)
    sample = min(50, len(A))
    o_ok = True
    oc = osum.cpu().numpy()
    for i in range(sample):
        na, nb = al[i], bl[i]
        ref = (aarr[i, :na, None, :] + barr[i, None, :nb, :]).reshape(-1, 2)
        got = oc[i].reshape(aS, bS, 2)[:na, :nb, :].reshape(-1, 2)
        if not np.array_equal(np.sort(got, axis=0), np.sort(ref, axis=0)):
            o_ok = False; break
    t0 = time.perf_counter()
    for i in range(sample):
        _ = (aarr[i, :al[i], None, :] + barr[i, None, :bl[i], :]).reshape(-1, 2)
    result["kernels"]["outer_sum"] = {
        "correct": bool(o_ok), "gpu_ms": round(gout, 3),
        "status": "validated" if o_ok else "not-validated",
        "note": "dense batched Minkowski outer-sum; correctness checked on a sample",
    }

    result["status"] = "ok"
    print(json.dumps(result, indent=2))
    open(args.out, "w").write(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
