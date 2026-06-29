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
use std::collections::{BinaryHeap, HashMap};

use mbor_core::graph::{Cost, Graph};
use mbor_core::label_setting::pareto_from;
use mbor_core::pareto::{minkowski_sum, pareto_filter};

/// Contiguous block partition: node `i` goes to fragment `(i * k) / n`.
/// Deterministic and trivial to reproduce in any language (used as a stand-in
/// for KaHIP min-cut; MBOR is exact, so the partition does not affect results).
pub fn contiguous_partition(n: usize, k: usize) -> Vec<u32> {
    assert!(k >= 1 && k <= n.max(1));
    (0..n).map(|i| ((i * k) / n) as u32).collect()
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

        // Per-fragment induced subgraphs + FPPV (forward and reverse).
        let mut frag_g2l: Vec<Vec<u32>> = Vec::with_capacity(k);
        let mut frag_graph: Vec<Graph> = Vec::with_capacity(k);
        let mut b2node: Vec<HashMap<u32, Vec<Vec<Cost>>>> = vec![HashMap::new(); k];
        let mut node2b: Vec<HashMap<u32, Vec<Vec<Cost>>>> = vec![HashMap::new(); k];

        for f in 0..k {
            let (gf, g2l) = graph.induced_subgraph(&frag_nodes[f]);
            let gf_rev = gf.reversed();
            for &b in &frag_boundary[f] {
                let bl = g2l[b as usize] as usize;
                // boundary -> node (forward) and node -> boundary (reverse).
                b2node[f].insert(b, pareto_from(&gf, bl));
                node2b[f].insert(b, pareto_from(&gf_rev, bl));
            }
            frag_g2l.push(g2l);
            frag_graph.push(gf);
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

        // BPPV: all-pairs Pareto over the boundary multigraph.
        let mut bppv: Vec<Vec<Vec<Cost>>> = Vec::with_capacity(nb);
        for s in 0..nb {
            bppv.push(multigraph_pareto_from(&adj, s));
        }

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
}
