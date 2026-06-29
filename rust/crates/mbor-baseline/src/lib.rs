//! Exact compute-on-demand bi-objective single-pair baselines for the speedup
//! comparison against MBOR's precomputed retrieval.
//!
//! Both run the same lexicographic label-setting over the full graph and return
//! the complete Pareto frontier (sorted by c1). They differ only in pruning:
//!   * [`bod`] (bi-objective Dijkstra, Martins): per-node truncated dominance
//!     (`g2_min`) only.
//!   * [`boa_star`] (BOA*): adds the goal bound (`g2_min[dest]`), the
//!     dimensionality-reduction prune of Hernandez et al. 2023, so it settles
//!     strictly fewer labels.
//!
//! These are the compute-on-demand methods MBOR is meant to beat with
//! precomputation; they share `mbor-core`'s exactness guarantee.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use mbor_core::graph::{Cost, Graph};

/// Bi-objective Dijkstra (BOD): Martins label-setting with per-node `g2_min`
/// truncated dominance and no goal bound.
pub fn bod(graph: &Graph, o: usize, d: usize) -> Vec<Cost> {
    let n = graph.num_nodes();
    let mut g2 = vec![i64::MAX; n];
    let mut sols: Vec<Cost> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(i64, i64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0, 0, o)));
    while let Some(Reverse((c1, c2, u))) = heap.pop() {
        if c2 >= g2[u] {
            continue;
        }
        g2[u] = c2;
        if u == d {
            sols.push(Cost::new(c1, c2));
            continue;
        }
        for (v, w) in graph.neighbors(u) {
            let n2 = c2 + w.c2;
            if n2 >= g2[v] {
                continue;
            }
            heap.push(Reverse((c1 + w.c1, n2, v)));
        }
    }
    sols
}

/// Bi-objective A* with zero heuristic (BOA*): BOD plus the goal-bound prune
/// (`g2_min[dest]`), which discards any label that cannot improve on the best
/// second cost already reached at the destination.
pub fn boa_star(graph: &Graph, o: usize, d: usize) -> Vec<Cost> {
    let n = graph.num_nodes();
    let mut g2 = vec![i64::MAX; n];
    let mut sols: Vec<Cost> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(i64, i64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0, 0, o)));
    while let Some(Reverse((c1, c2, u))) = heap.pop() {
        if c2 >= g2[u] || c2 >= g2[d] {
            continue;
        }
        g2[u] = c2;
        if u == d {
            sols.push(Cost::new(c1, c2));
            continue;
        }
        for (v, w) in graph.neighbors(u) {
            let n2 = c2 + w.c2;
            if n2 >= g2[v] || n2 >= g2[d] {
                continue;
            }
            heap.push(Reverse((c1 + w.c1, n2, v)));
        }
    }
    sols
}
