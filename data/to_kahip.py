#!/usr/bin/env python3
"""Convert an MBOR map (`<n> <m>` then `<u> <v> <c1> <c2>`, 1-indexed) to the
undirected KaHIP/METIS graph format (first line `<n> <undirected_edges>`, then n
lines of sorted 1-indexed neighbours). Costs and direction are dropped: the
partitioner only needs the connectivity. Self-loops and parallel edges removed."""
import sys

mapf, outf = sys.argv[1], sys.argv[2]
with open(mapf) as f:
    n, _m = map(int, f.readline().split())
    adj = [set() for _ in range(n + 1)]  # 1-indexed
    for line in f:
        p = line.split()
        if len(p) < 2:
            continue
        u, v = int(p[0]), int(p[1])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
medges = sum(len(a) for a in adj[1:]) // 2
with open(outf, "w") as g:
    g.write(f"{n} {medges}\n")
    for i in range(1, n + 1):
        g.write(" ".join(map(str, sorted(adj[i]))) + "\n")
print(f"{outf}: n={n} undirected_edges={medges}")
