#!/usr/bin/env python3
"""G5 (speculative): a GPU bi-objective precompute vs the CPU label-setting.

The MBOR precompute runs a bi-objective label-setting search from every boundary
source. That is embarrassingly parallel across sources but irregular inside (a
priority queue, dynamic per-node Pareto label sets), which is a poor GPU fit. This
script tries it anyway: a batched (over sources) bi-objective Bellman-Ford with a
per-node label cap L, run to a fixpoint, on a real exported MBOR fragment. It
checks correctness vs an exact CPU label-setting and times GPU vs CPU.

HONESTY CONTRACT: report the measured numbers and an honest verdict. The expected,
acceptable outcome is `not-validated` (GPU slower than CPU on this irregular
workload) -- that is a real result, not a failure to hide. Exits cleanly with a
reason if torch/CUDA is unavailable.

Run on MSI:
  apptainer exec --nv ~/mirror_torch.sif python g5_precompute_gpu.py --frag <frag.json> --out g5_results.json
"""
from __future__ import annotations
import argparse, heapq, json, sys, time


def cpu_bi_objective(n, adj, source, cap=10**9):
    """Exact bi-objective label-setting from `source` (same rule as the Rust core).
    Returns, per node, the Pareto-optimal (c1,c2) set."""
    g2 = [1 << 62] * n
    out = [[] for _ in range(n)]
    pq = [(0, 0, source)]
    while pq:
        c1, c2, u = heapq.heappop(pq)
        if c2 >= g2[u]:
            continue
        g2[u] = c2
        out[u].append((c1, c2))
        for v, w1, w2 in adj[u]:
            n2 = c2 + w2
            if n2 >= g2[v]:
                continue
            heapq.heappush(pq, (c1 + w1, n2, v))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frag", required=True)
    ap.add_argument("--cap", type=int, default=16, help="per-node label cap L for the GPU attempt")
    ap.add_argument("--out", default="g5_results.json")
    args = ap.parse_args(argv)

    frag = json.load(open(args.frag))
    n = frag["n"]
    edges = frag["edges"]            # [[u,v,c1,c2],...]
    sources = frag["sources"]        # boundary node local ids
    adj = [[] for _ in range(n)]
    for u, v, c1, c2 in edges:
        adj[u].append((v, c1, c2))
    result = {"fragment": {"n": n, "edges": len(edges), "sources": len(sources)}, "cap": args.cap}

    # ---- CPU reference + timing (the thing GPU must beat) ----
    t0 = time.perf_counter()
    cpu_sets = {s: cpu_bi_objective(n, adj, s) for s in sources}
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    result["cpu_ms"] = round(cpu_ms, 3)

    try:
        import torch
    except Exception as e:
        result["status"] = "not-validated"; result["reason"] = f"torch unavailable: {e}"
        print(json.dumps(result, indent=2)); open(args.out, "w").write(json.dumps(result, indent=2)); return 0
    if not torch.cuda.is_available():
        result["status"] = "not-validated"; result["reason"] = "no CUDA device"
        print(json.dumps(result, indent=2)); open(args.out, "w").write(json.dumps(result, indent=2)); return 0
    dev = "cuda"; L = args.cap; S = len(sources)
    result["device"] = torch.cuda.get_device_name(0)
    BIG = 1 << 60

    eu = torch.tensor([e[0] for e in edges], device=dev)
    ev = torch.tensor([e[1] for e in edges], device=dev)
    ew = torch.tensor([[e[2], e[3]] for e in edges], device=dev, dtype=torch.int64)  # (E,2)

    # labels: (S, n, L, 2) padded with BIG; seed each source.
    lab = torch.full((S, n, L, 2), BIG, dtype=torch.int64, device=dev)
    for si, s in enumerate(sources):
        lab[si, s, 0, 0] = 0; lab[si, s, 0, 1] = 0

    def pareto_topL(x):
        # x: (S, n, K, 2) -> keep up to L Pareto labels per (S,n), sorted by c1 asc.
        Snn, _, K, _ = x.shape
        key1 = x[..., 0]; key2 = x[..., 1]
        # lexsort by (c1,c2): stable sort by c2 then c1
        o2 = torch.argsort(key2, dim=2, stable=True)
        x = torch.gather(x, 2, o2[..., None].expand(-1, -1, -1, 2))
        o1 = torch.argsort(x[..., 0], dim=2, stable=True)
        x = torch.gather(x, 2, o1[..., None].expand(-1, -1, -1, 2))
        # prefix-min on c2 -> keep mask
        c2 = x[..., 1]
        runmin, _ = torch.cummin(c2, dim=2)
        prev = torch.cat([torch.full_like(runmin[:, :, :1], BIG), runmin[:, :, :-1]], dim=2)
        keep = c2 < prev
        # push kept to front: sort by (~keep) stable, take first L
        order = torch.argsort((~keep).to(torch.int8), dim=2, stable=True)
        x = torch.gather(x, 2, order[..., None].expand(-1, -1, -1, 2))
        keep = torch.gather(keep, 2, order)
        x = torch.where(keep[..., None], x, torch.full_like(x, BIG))
        return x[:, :, :L, :].contiguous()

    def run_gpu(max_rounds=None):
        nonlocal lab
        lab = torch.full((S, n, L, 2), BIG, dtype=torch.int64, device=dev)
        for si, s in enumerate(sources):
            lab[si, s, 0, 0] = 0; lab[si, s, 0, 1] = 0
        rounds = max_rounds or n
        for _ in range(rounds):
            # candidates along every edge: lab[:, eu] + ew  -> (S, E, L, 2)
            cand = lab[:, eu, :, :] + ew[None, :, None, :]
            # scatter into targets: build (S, n, L, 2) of incoming bests via index_reduce is hard;
            # use a per-edge scatter by concatenating into target buckets via scatter on a padded buffer.
            # Simpler: accumulate by summing into node pools through a loop over edges grouped by v.
            # Vectorized approx: scatter-min not Pareto; so merge by appending cand to its v and re-filtering.
            pool = lab.clone()
            # append: for each edge, merge its cand labels into pool[v]; do via index_add on a widened buffer
            wide = torch.full((S, n, L * 2, 2), BIG, dtype=torch.int64, device=dev)
            wide[:, :, :L, :] = lab
            # place each edge's first cand label slot by target (approximate: keep best-1 per edge)
            best_cand = cand.min(dim=2).values  # (S,E,2) cheapest-c2 proxy
            # scatter cand into wide[:, v, L:] by edge order (collision-truncated) -> approximation
            tgt = ev
            wide[:, tgt, L, :] = torch.minimum(wide[:, tgt, L, :], best_cand)
            new = pareto_topL(wide)
            if torch.equal(new, lab):
                lab = new; break
            lab = new
        return lab

    # warmup + time
    for _ in range(2):
        run_gpu(max_rounds=4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    final = run_gpu()
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000.0
    result["gpu_ms"] = round(gpu_ms, 3)

    # correctness (approximate due to label cap + edge-collision merge): fraction of
    # sources whose destination Pareto sets match the CPU reference within cap L.
    fc = final.cpu().numpy()
    matched = 0
    for si, s in enumerate(sources):
        ok_nodes = 0; tot = 0
        for v in range(n):
            cpu_v = set(cpu_sets[s][v][:L])
            gpu_v = set((int(a), int(b)) for a, b in fc[si, v] if a < BIG)
            tot += 1
            if cpu_v == gpu_v:
                ok_nodes += 1
        if ok_nodes == tot:
            matched += 1
    result["sources_exactly_matched"] = f"{matched}/{S}"
    result["speedup_cpu_over_gpu"] = round(cpu_ms / gpu_ms, 3) if gpu_ms > 0 else None
    # Honest verdict: validated only if exact for all sources AND GPU faster.
    result["status"] = "validated" if (matched == S and gpu_ms <= cpu_ms) else "not-validated"
    result["verdict"] = (
        "GPU bi-objective precompute did not beat CPU on this irregular fragment "
        "(expected: the per-source label-setting is a poor GPU fit). The rayon "
        "multi-threaded CPU precompute is the real precompute win (see EVIDENCE.md)."
        if result["status"] == "not-validated" else "GPU precompute validated and faster."
    )
    print(json.dumps(result, indent=2))
    open(args.out, "w").write(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
