//! MBOR precomputation: the Multi-level Encoded Pareto Frontier View (MEPFV).
//!
//! Given a graph and a fragment partition, this builds:
//!   * the **boundary nodes** (endpoints of cross-fragment edges),
//!   * the **fragment Pareto path views (FPPV)**: within-fragment Pareto sets
//!     from every node to each boundary node and from each boundary node to
//!     every node (Algorithm 2 restricted to a fragment),
//!   * the **boundary multigraph** `G^b`: boundary nodes connected by local
//!     multi-edges (within-fragment boundary-to-boundary Pareto sets) and
//!     boundary edges (original cross-fragment edges), and
//!   * the **boundary Pareto path view (BPPV)**: all-pairs Pareto sets over the
//!     boundary multigraph.
//!
//! `Mepfv::query` is the basic online retrieval (Algorithm 3): it assembles the
//! complete Pareto frontier for an origin-destination pair by combining
//! FPPV(o -> oBN), BPPV(oBN -> dBN), and FPPV(dBN -> d) over all boundary-node
//! pairs (plus the within-fragment direct paths when o and d share a fragment).
//!
//! Correctness is checked against `mbor_core`'s exact full-graph search.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

use mbor_core::graph::{Cost, Graph};
use mbor_core::label_setting::pareto_from;
use mbor_core::pareto::{insert_nondominated, minkowski_sum, pareto_filter};
use rayon::prelude::*;

/// Contiguous block partition: node `i` goes to fragment `(i * k) / n`.
/// Deterministic and trivial to reproduce in any language (used as a stand-in
/// for KaHIP min-cut; MBOR is exact, so the partition does not affect results).
pub fn contiguous_partition(n: usize, k: usize) -> Vec<u32> {
    assert!(k >= 1 && k <= n.max(1));
    (0..n).map(|i| ((i * k) / n) as u32).collect()
}

/// Multi-source BFS (Voronoi) region-growing partition: `k` seeds spread across
/// the node-id range grow outward simultaneously, each node joining the seed
/// that reaches it first. Produces compact, connected fragments with far fewer
/// boundary nodes than a contiguous split (a dependency-free stand-in for KaHIP
/// min-cut). Deterministic. MBOR is exact, so the partition affects only speed.
pub fn bfs_partition(graph: &Graph, k: usize) -> Vec<u32> {
    let n = graph.num_nodes();
    assert!(k >= 1 && k <= n.max(1));
    // Undirected adjacency for region growing.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for u in 0..n {
        for (v, _) in graph.neighbors(u) {
            adj[u].push(v as u32);
            adj[v].push(u as u32);
        }
    }
    let mut part = vec![u32::MAX; n];
    let mut q: VecDeque<u32> = VecDeque::new();
    for fid in 0..k {
        let s = (fid * n) / k;
        if part[s] == u32::MAX {
            part[s] = fid as u32;
            q.push_back(s as u32);
        }
    }
    while let Some(u) = q.pop_front() {
        let f = part[u as usize];
        for &v in &adj[u as usize] {
            if part[v as usize] == u32::MAX {
                part[v as usize] = f;
                q.push_back(v);
            }
        }
    }
    for p in part.iter_mut() {
        if *p == u32::MAX {
            *p = 0; // isolated nodes -> fragment 0
        }
    }
    part
}

/// Pareto label-setting over a boundary multigraph: `adj[u]` lists
/// `(neighbor, cost_set)` multi-edges. Returns, for each node, the Pareto set of
/// costs from `source` (sorted by c1).
fn multigraph_pareto_from(adj: &[Vec<(usize, Vec<Cost>)>], source: usize) -> Vec<Vec<Cost>> {
    let n = adj.len();
    let mut g2_min = vec![i64::MAX; n];
    let mut out: Vec<Vec<Cost>> = vec![Vec::new(); n];
    let mut heap: BinaryHeap<Reverse<(i64, i64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0, 0, source)));
    while let Some(Reverse((c1, c2, u))) = heap.pop() {
        if c2 >= g2_min[u] {
            continue;
        }
        g2_min[u] = c2;
        out[u].push(Cost::new(c1, c2));
        for (v, costs) in &adj[u] {
            for w in costs {
                let n2 = c2 + w.c2;
                if n2 >= g2_min[*v] {
                    continue;
                }
                heap.push(Reverse((c1 + w.c1, n2, *v)));
            }
        }
    }
    out
}

/// The precomputed Multi-level Encoded Pareto Frontier View.
pub struct Mepfv {
    n: usize,
    part: Vec<u32>,
    /// global node id -> local index within its own fragment.
    frag_g2l: Vec<Vec<u32>>, // [fragment][global] = local or u32::MAX
    frag_graph: Vec<Graph>,       // forward induced subgraph per fragment
    frag_boundary: Vec<Vec<u32>>, // boundary node global ids per fragment
    /// FPPV boundary -> node: b2node[f][global_boundary][local_node] Pareto set.
    b2node: Vec<HashMap<u32, Vec<Vec<Cost>>>>,
    /// FPPV node -> boundary: node2b[f][global_boundary][local_node] Pareto set.
    node2b: Vec<HashMap<u32, Vec<Vec<Cost>>>>,
    /// global node id -> boundary index, or -1.
    bidx: Vec<i32>,
    /// BPPV all-pairs over the boundary multigraph: bppv[s][t] Pareto set.
    bppv: Vec<Vec<Vec<Cost>>>,
}

impl Mepfv {
    /// Build the MEPFV for `graph` under fragment assignment `part`.
    pub fn build(graph: &Graph, part: Vec<u32>) -> Mepfv {
        let n = graph.num_nodes();
        assert_eq!(part.len(), n, "partition length must equal node count");
        let k = (*part.iter().max().unwrap_or(&0) as usize) + 1;

        // Group nodes by fragment.
        let mut frag_nodes: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (g, &f) in part.iter().enumerate() {
            frag_nodes[f as usize].push(g as u32);
        }

        // Boundary nodes: endpoints of any cross-fragment edge.
        let mut is_boundary = vec![false; n];
        for u in 0..n {
            for (v, _) in graph.neighbors(u) {
                if part[u] != part[v] {
                    is_boundary[u] = true;
                    is_boundary[v] = true;
                }
            }
        }
        let mut boundary: Vec<u32> = (0..n as u32).filter(|&g| is_boundary[g as usize]).collect();
        boundary.sort_unstable();
        let mut bidx = vec![-1i32; n];
        for (i, &g) in boundary.iter().enumerate() {
            bidx[g as usize] = i as i32;
        }
        let mut frag_boundary: Vec<Vec<u32>> = vec![Vec::new(); k];
        for &g in &boundary {
            frag_boundary[part[g as usize] as usize].push(g);
        }

        // Per-fragment induced subgraphs + FPPV (forward and reverse). Fragments
        // are independent, so the FPPV encode runs in parallel across fragments
        // (rayon); set RAYON_NUM_THREADS=1 for the single-threaded baseline.
        type FppvPart = (
            Vec<u32>,
            Graph,
            HashMap<u32, Vec<Vec<Cost>>>,
            HashMap<u32, Vec<Vec<Cost>>>,
        );
        let frag_results: Vec<FppvPart> = (0..k)
            .into_par_iter()
            .map(|f| {
                let (gf, g2l) = graph.induced_subgraph(&frag_nodes[f]);
                let gf_rev = gf.reversed();
                let mut b2: HashMap<u32, Vec<Vec<Cost>>> = HashMap::new();
                let mut n2: HashMap<u32, Vec<Vec<Cost>>> = HashMap::new();
                for &b in &frag_boundary[f] {
                    let bl = g2l[b as usize] as usize;
                    b2.insert(b, pareto_from(&gf, bl)); // boundary -> node
                    n2.insert(b, pareto_from(&gf_rev, bl)); // node -> boundary
                }
                (g2l, gf, b2, n2)
            })
            .collect();
        let mut frag_g2l: Vec<Vec<u32>> = Vec::with_capacity(k);
        let mut frag_graph: Vec<Graph> = Vec::with_capacity(k);
        let mut b2node: Vec<HashMap<u32, Vec<Vec<Cost>>>> = Vec::with_capacity(k);
        let mut node2b: Vec<HashMap<u32, Vec<Vec<Cost>>>> = Vec::with_capacity(k);
        for (g2l, gf, b2, n2) in frag_results {
            frag_g2l.push(g2l);
            frag_graph.push(gf);
            b2node.push(b2);
            node2b.push(n2);
        }

        // Boundary multigraph adjacency over boundary indices.
        let nb = boundary.len();
        let mut adj: Vec<Vec<(usize, Vec<Cost>)>> = vec![Vec::new(); nb];
        // Local multi-edges: within-fragment boundary -> boundary Pareto sets.
        for f in 0..k {
            for &b1 in &frag_boundary[f] {
                let from = bidx[b1 as usize] as usize;
                let table = &b2node[f][&b1];
                for &b2 in &frag_boundary[f] {
                    if b1 == b2 {
                        continue;
                    }
                    let b2l = frag_g2l[f][b2 as usize] as usize;
                    let costs = &table[b2l];
                    if !costs.is_empty() {
                        adj[from].push((bidx[b2 as usize] as usize, costs.clone()));
                    }
                }
            }
        }
        // Boundary edges: original cross-fragment edges.
        for u in 0..n {
            for (v, c) in graph.neighbors(u) {
                if part[u] != part[v] {
                    adj[bidx[u] as usize].push((bidx[v] as usize, vec![c]));
                }
            }
        }

        // BPPV: all-pairs Pareto over the boundary multigraph. Each source is
        // independent -> parallelized across boundary nodes (the heaviest part
        // of precompute when there are many boundary nodes).
        let bppv: Vec<Vec<Vec<Cost>>> = (0..nb)
            .into_par_iter()
            .map(|s| multigraph_pareto_from(&adj, s))
            .collect();

        Mepfv {
            n,
            part,
            frag_g2l,
            frag_graph,
            frag_boundary,
            b2node,
            node2b,
            bidx,
            bppv,
        }
    }

    /// Convenience: build with a contiguous `k`-fragment partition.
    pub fn build_contiguous(graph: &Graph, k: usize) -> Mepfv {
        let part = contiguous_partition(graph.num_nodes(), k);
        Mepfv::build(graph, part)
    }

    pub fn num_boundary(&self) -> usize {
        self.bidx.iter().filter(|&&b| b >= 0).count()
    }

    /// Online retrieval (Algorithm 3, Basic): the complete Pareto frontier of
    /// `(c1, c2)` costs from `o` to `d`, sorted by c1.
    pub fn query(&self, o: usize, d: usize) -> Vec<Cost> {
        assert!(o < self.n && d < self.n);
        if o == d {
            return vec![Cost::ZERO];
        }
        let fo = self.part[o] as usize;
        let fd = self.part[d] as usize;
        let mut candidates: Vec<Cost> = Vec::new();

        // Within-fragment direct paths when o and d share a fragment.
        if fo == fd {
            let ol = self.frag_g2l[fo][o] as usize;
            let dl = self.frag_g2l[fo][d] as usize;
            let direct = pareto_from(&self.frag_graph[fo], ol);
            candidates.extend_from_slice(&direct[dl]);
        }

        // Boundary-routed paths: o -> oBN (FPPV) + oBN -> dBN (BPPV) + dBN -> d (FPPV).
        let ol = self.frag_g2l[fo][o] as usize;
        let dl = self.frag_g2l[fd][d] as usize;
        for &obn in &self.frag_boundary[fo] {
            let a = &self.node2b[fo][&obn][ol]; // Pareto(o -> oBN)
            if a.is_empty() {
                continue;
            }
            let si = self.bidx[obn as usize] as usize;
            for &dbn in &self.frag_boundary[fd] {
                let c = &self.b2node[fd][&dbn][dl]; // Pareto(dBN -> d)
                if c.is_empty() {
                    continue;
                }
                let ti = self.bidx[dbn as usize] as usize;
                let b = &self.bppv[si][ti]; // Pareto(oBN -> dBN)
                if b.is_empty() {
                    continue;
                }
                let ab = minkowski_sum(a, b);
                let abc = minkowski_sum(&ab, c);
                candidates.extend(abc);
            }
        }

        pareto_filter(candidates)
    }

    /// Online retrieval (Algorithm 4, Adv): the SAME exact frontier as `query`,
    /// but with two-dimensional cost-interval pruning. For each boundary-node
    /// pair the route's ideal lower-left corner `(min c1, min c2)` is the best
    /// any of its paths can do; if that corner is already dominated by the
    /// frontier found so far, every path through the pair is dominated, so its
    /// Minkowski combination is skipped. Pairs are processed ideal-corner-first
    /// so the frontier tightens early. Returns the frontier plus pruning stats.
    pub fn query_adv(&self, o: usize, d: usize) -> (Vec<Cost>, AdvStats) {
        assert!(o < self.n && d < self.n);
        let mut stats = AdvStats::default();
        if o == d {
            return (vec![Cost::ZERO], stats);
        }
        let fo = self.part[o] as usize;
        let fd = self.part[d] as usize;
        let ol = self.frag_g2l[fo][o] as usize;
        let dl = self.frag_g2l[fd][d] as usize;

        // Seed the frontier with within-fragment direct paths.
        let mut front: Vec<Cost> = Vec::new();
        if fo == fd {
            let direct = pareto_from(&self.frag_graph[fo], ol);
            for &c in &direct[dl] {
                insert_nondominated(&mut front, c);
            }
        }

        // Candidate (oBN, dBN) pairs: segment frontiers + ideal corner.
        struct Pair<'a> {
            a: &'a [Cost],
            b: &'a [Cost],
            c: &'a [Cost],
            c1min: i64,
            c2min: i64,
        }
        let mut pairs: Vec<Pair> = Vec::new();
        for &obn in &self.frag_boundary[fo] {
            let a = &self.node2b[fo][&obn][ol];
            if a.is_empty() {
                continue;
            }
            let si = self.bidx[obn as usize] as usize;
            for &dbn in &self.frag_boundary[fd] {
                let c = &self.b2node[fd][&dbn][dl];
                if c.is_empty() {
                    continue;
                }
                let ti = self.bidx[dbn as usize] as usize;
                let b = &self.bppv[si][ti];
                if b.is_empty() {
                    continue;
                }
                // a, b, c are sorted by c1 asc / c2 desc: [0] = min c1, last = min c2.
                let c1min = a[0].c1 + b[0].c1 + c[0].c1;
                let c2min = a[a.len() - 1].c2 + b[b.len() - 1].c2 + c[c.len() - 1].c2;
                pairs.push(Pair {
                    a,
                    b,
                    c,
                    c1min,
                    c2min,
                });
            }
        }
        stats.pairs_total = pairs.len();
        pairs.sort_by(|x, y| x.c1min.cmp(&y.c1min).then(x.c2min.cmp(&y.c2min)));

        for p in &pairs {
            if front.iter().any(|f| f.c1 <= p.c1min && f.c2 <= p.c2min) {
                stats.pairs_pruned += 1;
                continue;
            }
            let ab = minkowski_sum(p.a, p.b);
            let abc = minkowski_sum(&ab, p.c);
            stats.combinations += 1;
            for c in abc {
                insert_nondominated(&mut front, c);
            }
        }

        (pareto_filter(front), stats)
    }

    /// Just the Adv frontier (for parity checks).
    pub fn query_adv_costs(&self, o: usize, d: usize) -> Vec<Cost> {
        self.query_adv(o, d).0
    }

    /// Export the boundary-pair segment cost-sets `(A, B, C)` for a query, where
    /// `A = FPPV(o->oBN)`, `B = BPPV(oBN->dBN)`, `C = FPPV(dBN->d)`. This is the
    /// batched Minkowski-combine workload the Triton kernels accelerate.
    pub fn export_pairs(&self, o: usize, d: usize) -> Vec<(Vec<Cost>, Vec<Cost>, Vec<Cost>)> {
        let fo = self.part[o] as usize;
        let fd = self.part[d] as usize;
        let ol = self.frag_g2l[fo][o] as usize;
        let dl = self.frag_g2l[fd][d] as usize;
        let mut out = Vec::new();
        for &obn in &self.frag_boundary[fo] {
            let a = &self.node2b[fo][&obn][ol];
            if a.is_empty() {
                continue;
            }
            let si = self.bidx[obn as usize] as usize;
            for &dbn in &self.frag_boundary[fd] {
                let c = &self.b2node[fd][&dbn][dl];
                if c.is_empty() {
                    continue;
                }
                let ti = self.bidx[dbn as usize] as usize;
                let b = &self.bppv[si][ti];
                if b.is_empty() {
                    continue;
                }
                out.push((a.clone(), b.clone(), c.clone()));
            }
        }
        out
    }

    /// Export the largest fragment as `(num_nodes, edges, boundary_source_locals)`
    /// for the speculative GPU precompute (a self-contained subgraph + the local
    /// indices of its boundary nodes to run bi-objective search from).
    pub fn largest_fragment(&self) -> (usize, Vec<(u32, u32, i64, i64)>, Vec<u32>) {
        let f = (0..self.frag_graph.len())
            .max_by_key(|&f| self.frag_graph[f].num_nodes())
            .unwrap_or(0);
        let g = &self.frag_graph[f];
        let edges: Vec<(u32, u32, i64, i64)> = g
            .edges()
            .map(|(u, v, c)| (u as u32, v as u32, c.c1, c.c2))
            .collect();
        let srcs: Vec<u32> = self.frag_boundary[f]
            .iter()
            .map(|&b| self.frag_g2l[f][b as usize])
            .collect();
        (g.num_nodes(), edges, srcs)
    }
}

/// Pruning statistics from `Mepfv::query_adv`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdvStats {
    pub pairs_total: usize,
    pub pairs_pruned: usize,
    pub combinations: usize,
}

/// Load a partition file (one fragment id per line in node order 0..n-1), e.g.
/// a KaHIP `kaffpaIndex.txt` or the synthesized stand-in.
pub fn load_partition(path: &str) -> Result<Vec<u32>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    s.split_whitespace()
        .map(|t| t.parse::<u32>().map_err(|e| e.to_string()))
        .collect()
}
