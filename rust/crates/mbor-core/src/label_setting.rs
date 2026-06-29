//! Bi-objective single-pair label-setting search (Algorithm 2 of the paper; the
//! BOA* / Bi-Objective-Dijkstra family). Returns the complete Pareto frontier
//! of `(c1, c2)` costs from a source to a destination, each with a path.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::graph::{Cost, Graph};

#[derive(Clone)]
struct Label {
    cost: Cost,
    node: u32,
    /// Index into the label arena, or `-1` for the root label at the source.
    parent: i32,
}

/// Heap entry ordered lexicographically by `(c1, c2)`. `BinaryHeap` is a
/// max-heap, so the comparison is reversed to pop the smallest pair first.
#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapItem {
    c1: i64,
    c2: i64,
    label: u32,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.c1.cmp(&self.c1).then(other.c2.cmp(&self.c2))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A Pareto-optimal solution: its cost and the reconstructed path as a sequence
/// of 0-indexed node ids from source to destination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoPath {
    pub cost: Cost,
    pub path: Vec<u32>,
}

/// Find the complete Pareto frontier from `source` to `dest`.
///
/// Lexicographic label-setting with the `g2_min` truncated-dominance prune
/// (BOA*): labels are popped in `(c1, c2)` order, a node is settled only when a
/// label improves on the best `c2` settled there, and any label whose `c2`
/// cannot beat the best `c2` already reached at `dest` is pruned. Because `c1`
/// is non-decreasing at extraction, the first settle of a node at a given `c2`
/// is Pareto-optimal.
pub fn pareto_search(graph: &Graph, source: usize, dest: usize) -> Vec<ParetoPath> {
    let n = graph.num_nodes();
    let mut g2_min = vec![i64::MAX; n];
    let mut labels: Vec<Label> = Vec::new();
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut solutions: Vec<u32> = Vec::new();

    labels.push(Label {
        cost: Cost::ZERO,
        node: source as u32,
        parent: -1,
    });
    heap.push(HeapItem {
        c1: 0,
        c2: 0,
        label: 0,
    });

    while let Some(item) = heap.pop() {
        let li = item.label as usize;
        let (cost, u) = {
            let l = &labels[li];
            (l.cost, l.node as usize)
        };

        // Truncated dominance: prune labels already dominated at `u`, and any
        // label that cannot improve on the best cost reached at `dest`.
        if cost.c2 >= g2_min[u] || cost.c2 >= g2_min[dest] {
            continue;
        }
        g2_min[u] = cost.c2;

        if u == dest {
            solutions.push(li as u32);
            continue;
        }

        for (v, w) in graph.neighbors(u) {
            let nc = cost + w;
            if nc.c2 >= g2_min[v] || nc.c2 >= g2_min[dest] {
                continue;
            }
            let idx = labels.len() as u32;
            labels.push(Label {
                cost: nc,
                node: v as u32,
                parent: li as i32,
            });
            heap.push(HeapItem {
                c1: nc.c1,
                c2: nc.c2,
                label: idx,
            });
        }
    }

    let mut out: Vec<ParetoPath> = solutions
        .iter()
        .map(|&s| {
            let mut path = Vec::new();
            let mut cur = s as i32;
            while cur >= 0 {
                let l = &labels[cur as usize];
                path.push(l.node);
                cur = l.parent;
            }
            path.reverse();
            ParetoPath {
                cost: labels[s as usize].cost,
                path,
            }
        })
        .collect();
    out.sort_by(|a, b| a.cost.c1.cmp(&b.cost.c1).then(a.cost.c2.cmp(&b.cost.c2)));
    out
}

/// The Pareto-optimal cost pairs from `source` to `dest`, sorted by `c1`.
pub fn pareto_costs(graph: &Graph, source: usize, dest: usize) -> Vec<Cost> {
    pareto_search(graph, source, dest)
        .into_iter()
        .map(|p| p.cost)
        .collect()
}
