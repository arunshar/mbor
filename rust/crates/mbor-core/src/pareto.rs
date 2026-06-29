//! Pareto-set utilities: maintaining a non-dominated frontier and the Minkowski
//! sum of two frontiers (the combination step in MBOR's online retrieval, and
//! the target of the two-dimensional cost-interval pruning in MBOR-Adv).

use crate::graph::Cost;

/// Insert `c` into a frontier held as non-dominated cost pairs: drop any member
/// that `c` dominates, and reject `c` if an existing member dominates it (or
/// equals it). Returns `true` if `c` was added.
pub fn insert_nondominated(frontier: &mut Vec<Cost>, c: Cost) -> bool {
    for &existing in frontier.iter() {
        if existing == c || existing.dominates(c) {
            return false;
        }
    }
    frontier.retain(|&existing| !c.dominates(existing));
    frontier.push(c);
    true
}

/// Reduce an arbitrary multiset of cost pairs to its Pareto frontier, returned
/// sorted by `c1` ascending (so `c2` is strictly decreasing along the result).
pub fn pareto_filter(mut costs: Vec<Cost>) -> Vec<Cost> {
    costs.sort_by(|a, b| a.c1.cmp(&b.c1).then(a.c2.cmp(&b.c2)));
    let mut out: Vec<Cost> = Vec::new();
    let mut best_c2 = i64::MAX;
    for c in costs {
        // With c1 non-decreasing, a point is non-dominated iff its c2 strictly
        // improves on the best c2 seen so far (ties in c1 resolved by the sort).
        if c.c2 < best_c2 {
            out.push(c);
            best_c2 = c.c2;
        }
    }
    out
}

/// Minkowski sum of two frontiers, reduced to the Pareto frontier. This is the
/// combinatorial `O(m*n)` combination MBOR's 2DCI pruning is designed to avoid
/// when the result would be dominated anyway.
pub fn minkowski_sum(a: &[Cost], b: &[Cost]) -> Vec<Cost> {
    let mut all = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            all.push(x + y);
        }
    }
    pareto_filter(all)
}
